"""Оркестрация: happy-path разметки, КМ только из отчёта, роли документов."""
from __future__ import annotations


from conftest import layout_answer, metric_answer
from helpers import FakeClient, make_docx, make_package, make_workbook
from laim_basket.journal import Journal
from laim_basket.llm import tasks

ROWS = [["q", "a", "m", "alt"], ["в1", "о1", 1, 1], ["в2", "о2", 0, 0.8]]
# Заявленное значение, которого пересчёт по колонке C (0.5) не даёт.
REPORTED_MISMATCH = {"state": "declared", "value": 0.9, "raw": "0.9"}


def _context(tmp_path):
    package = make_package(tmp_path, {"Лист1": {"rows": ROWS}})
    return tasks.build_run_context(package)


def _mismatching(**overrides):
    return metric_answer(reported_value=REPORTED_MISMATCH, **overrides)


def _layout(tmp_path, journal):
    ctx = _context(tmp_path)
    outcome = tasks.run_layout(FakeClient([layout_answer()]), ctx, journal, "",
                               frozenset())
    return ctx, outcome


def test_context_assigns_document_ports_by_domain_names(tmp_path):
    package = tmp_path / "CI09000002_x"
    package.mkdir()
    make_workbook(package / "data.xlsx", {"Лист1": {"rows": ROWS}})
    make_docx(package / "отчёт_о_валидации_342293.docx", paragraphs=("КМ 0.5",))
    make_docx(package / "Отчет о разработке 342293.docx", paragraphs=("Колонки",))
    make_docx(package / "Инструкция по разметке.docx", paragraphs=("Правила",))
    ctx = tasks.build_run_context(package)
    assert [document["port"] for document in ctx.documents] == [
        "validation_report", "development_report", "assessor_instruction"]
    assert ctx.basket_id == "CI09000002"


def test_layout_happy_path(tmp_path):
    _ctx, outcome = _layout(tmp_path, Journal())
    assert len(outcome.frame) == 2
    assert outcome.conversion["umr_validation"]["status"] == "passed"


def test_metric_match_needs_single_call(tmp_path):
    journal = Journal()
    ctx, outcome = _layout(tmp_path, journal)
    client = FakeClient([metric_answer()])

    _plan, metric, _published = tasks.run_metric(client, ctx, outcome, journal)

    assert metric["reconciliation"]["status"] == "match"
    assert client.calls == 1
    assert journal.warnings == []


def test_metric_not_declared_publishes_no_value(tmp_path):
    # Дефект артефактов сигнализирует контракт main.py (official_baseline_missing);
    # здесь фиксируется, что пересчёт не публикуется как значение КМ.
    journal = Journal()
    ctx, outcome = _layout(tmp_path, journal)
    absent = metric_answer(
        reported_value={"state": "not_declared", "value": None, "raw": None})

    _plan, metric, published = tasks.run_metric(
        FakeClient([absent]), ctx, outcome, journal)

    assert metric["main_metric"]["value"] is None
    assert metric["main_metric"]["recomputed_value"] == 0.5  # информационно
    assert "main_metric" in published.frame.columns  # корзина со score-колонками


def test_metric_mismatch_is_informational_only(tmp_path):
    journal = Journal()
    package = make_package(tmp_path, {"Лист1": {"rows": ROWS}},
                           validation=("Ключевая метрика Accuracy равна 0.9",))
    ctx = tasks.build_run_context(package)
    outcome = tasks.run_layout(FakeClient([layout_answer()]), ctx, journal, "",
                               frozenset())
    # Заявлено 0.9, пересчёт по C даёт 0.5; второй ответ не должен понадобиться.
    client = FakeClient([_mismatching(), _mismatching()])

    _plan, metric, _published = tasks.run_metric(client, ctx, outcome, journal)

    assert float(metric["main_metric"]["value"]) == 0.9
    assert metric["reconciliation"]["status"] == "mismatch"  # информационно
    assert client.calls == 1
    assert journal.warnings == []


def test_metric_uncited_reported_value_goes_to_repair(tmp_path):
    journal = Journal()
    ctx, outcome = _layout(tmp_path, journal)      # отчёт: «…равна 0.5»
    uncited = metric_answer(
        reported_value={"state": "declared", "value": 0.9, "raw": "0.9"})
    client = FakeClient([uncited, metric_answer()])

    _plan, metric, _published = tasks.run_metric(client, ctx, outcome, journal)

    assert client.labels == ["metric_turn1", "metric_turn2"]  # repair, не сверка
    assert float(metric["main_metric"]["value"]) == 0.5

def test_layout_warns_when_session_source_rejected(tmp_path):
    journal = Journal()
    rows = [["s", "q", "a", "m"],
            ["d1", "в1", "о1", 1],
            [None, "в2", "о2", 0]]
    package = make_package(tmp_path, {"Лист1": {"rows": rows}})
    ctx = tasks.build_run_context(package)
    answer = layout_answer(
        roles={"query_id": None, "session_id": "A", "input_query": "B",
               "output_answer": "C", "scenario": None, "assessor_id": None,
               "reference_answers": []},
        grouping={"kind": "column", "column": "A"})

    outcome = tasks.run_layout(FakeClient([answer]), ctx, journal, "", frozenset())

    assert outcome.frame["session_id"].tolist() == [0, 0]
    assert any(w["code"] == "session_id_source_rejected" for w in journal.warnings)

def _blob_answer():
    return layout_answer(
        roles={"query_id": None, "session_id": None, "input_query": "A",
               "output_answer": None, "scenario": None, "assessor_id": None,
               "reference_answers": []},
        grouping={"kind": "blob_row", "column": None},
        dialogue_blob={"column": "A", "container": "plain_text",
                       "question_marker": "клиент", "answer_marker": "оператор"})


def test_undecodable_blob_rows_dropped_after_repair(tmp_path):
    journal = Journal()
    rows = [["dialog", "m"],
            ["клиент: привет оператор: здравствуйте", 1],
            ["протокол без маркеров", 0],
            ["клиент: вопрос оператор: ответ", 1]]
    package = make_package(tmp_path, {"Лист1": {"rows": rows}})
    ctx = tasks.build_run_context(package)
    answer = _blob_answer()
    client = FakeClient([answer, answer, answer])

    outcome = tasks.run_layout(client, ctx, journal, "", frozenset())

    assert outcome.frame["input_query"].tolist() == ["привет", "вопрос"]
    assert journal.dropped_rows["undecodable_dialogue_blob"] == [3]
    assert client.calls == 3  # строгость в repair-цикле сохранена


def test_blob_without_any_unrollable_row_still_fails(tmp_path):
    # Ни одной развёрнутой строки — это мусор или неверная разметка: падение.
    from laim_basket.errors import SpecError
    journal = Journal()
    rows = [["dialog", "m"], ["мусор", 1], ["ещё мусор", 0]]
    package = make_package(tmp_path, {"Лист1": {"rows": rows}})
    ctx = tasks.build_run_context(package)
    answer = _blob_answer()

    import pytest
    with pytest.raises(SpecError):
        tasks.run_layout(FakeClient([answer, answer, answer]), ctx, journal,
                         "", frozenset())

def test_grouped_blank_input_query_rows_dropped_after_repair(tmp_path):
    journal = Journal()
    rows = [["s", "q", "a", "m"],
            [1, "в1", "о1", 1],
            [None, None, "о2", None],
            [2, "в3", "о3", 0]]
    package = make_package(
        tmp_path, {"Лист1": {"rows": rows, "merges": ["A2:A3", "D2:D3"]}})
    ctx = tasks.build_run_context(package)
    answer = layout_answer(
        roles={"query_id": None, "session_id": "A", "input_query": "B",
               "output_answer": "C", "scenario": None, "assessor_id": None,
               "reference_answers": []},
        grouping={"kind": "merged_rows", "column": None})
    client = FakeClient([answer, answer, answer])

    outcome = tasks.run_layout(client, ctx, journal, "", frozenset())

    assert outcome.frame["input_query"].tolist() == ["в1", "в3"]
    assert journal.dropped_rows["blank_input_query"] == [3]
    assert client.calls == 3


def test_blank_group_rows_dropped_after_repair(tmp_path):
    journal = Journal()
    rows = [["s", "q", "a", "m"],
            [None, "в0", "о0", 1],
            ["d1", "в1", "о1", 1],
            ["d1", "в2", "о2", 0]]
    package = make_package(tmp_path, {"Лист1": {"rows": rows}})
    ctx = tasks.build_run_context(package)
    answer = layout_answer(
        roles={"query_id": None, "session_id": None, "input_query": "B",
               "output_answer": "C", "scenario": None, "assessor_id": None,
               "reference_answers": []},
        grouping={"kind": "column", "column": "A"})
    client = FakeClient([answer, answer, answer])

    outcome = tasks.run_layout(client, ctx, journal, "", frozenset())

    assert outcome.frame["input_query"].tolist() == ["в1", "в2"]
    assert journal.dropped_rows["blank_group"] == [2]

