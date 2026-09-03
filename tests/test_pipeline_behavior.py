"""Поведение конвейера: инварианты спеки на формах корзин + деградации."""
from __future__ import annotations

import ast


from conftest import layout_answer, metric_answer
from helpers import FakeClient, make_package
from laim_basket.pipeline import run_package


def test_flat_basket_meets_spec_invariants(tmp_path):
    package = make_package(tmp_path, {"Лист1": {"rows": [
        ["q", "a", "m"], ["в1", "о1", 1], ["в2", "о2", 0]]}})
    result = run_package(package, tmp_path / "out",
                          client=FakeClient([layout_answer(), metric_answer()]))
    frame = result.umr.frame
    for column in ("query_id", "input_query", "output_answer", "main_metric"):
        assert column in frame.columns
    assert result.status == "computed"
    assert result.report["contract_version"] == "laim-run-report.v1"
    assert result.report["km"]["reconciliation"] == "match"
    assert [stage["stage"] for stage in result.report["stages"][:2]] == ["read", "layout"]
    assert (tmp_path / "out" / "run_report.json").exists()
    assert (tmp_path / "out" / f"umr_{result.report['basket_id']}.xlsx").exists()


def test_merged_dialogue_publishes_triples(tmp_path):
    package = make_package(tmp_path, {"Лист1": {
        "rows": [["s", "q", "a", "m"],
                  [1, "в1", "о1", 1], [None, "в2", "о2", None],
                  [2, "в3", "о3", 0]],
        "merges": ["A2:A3", "D2:D3"]}})
    result = run_package(package, tmp_path / "out", client=FakeClient([
        layout_answer(roles={"query_id": None, "session_id": "A",
                              "input_query": "B", "output_answer": "C",
                              "scenario": None, "assessor_id": None,
                              "reference_answers": []},
                       grouping={"kind": "merged_rows", "column": None}),
        metric_answer(sources=[{"column_id": "D", "role": "final_score",
                                 "normalization": "numeric",
                                 "polarity": "direct"}])]))
    frame = result.umr.frame
    assert "dialogue" in frame.columns
    turns = ast.literal_eval(frame["dialogue"].iloc[0])
    assert len(turns) == 2 and len(turns[0]) == 3
    assert result.report["decisions"]["assessment_mode"] == "dialogue"


def test_blob_dialogue_expands_turns(tmp_path):
    blob = "['КЛИЕНТ: привет', 'АГЕНТ: здравствуйте', 'КЛИЕНТ: вопрос', 'АГЕНТ: ответ']"
    package = make_package(tmp_path, {"Лист1": {"rows": [
        ["messages", "m"], [blob, 1],
        ["['КЛИЕНТ: ещё', 'АГЕНТ: снова']", 0]]}})
    result = run_package(package, tmp_path / "out", client=FakeClient([
        layout_answer(roles={"query_id": None, "session_id": None,
                              "input_query": "A", "output_answer": None,
                              "scenario": None, "assessor_id": None,
                              "reference_answers": []},
                       grouping={"kind": "blob_row", "column": None},
                       dialogue_blob={"column": "A", "container": "python_list",
                                       "question_marker": "КЛИЕНТ",
                                       "answer_marker": "АГЕНТ"}),
        metric_answer(sources=[{"column_id": "B", "role": "final_score",
                                 "normalization": "numeric",
                                 "polarity": "direct"}])]))
    frame = result.umr.frame
    assert "dialogue" in frame.columns
    turns = ast.literal_eval(frame["dialogue"].iloc[0])
    assert [turn[1] for turn in turns] == ["привет", "вопрос"]


def test_classification_accuracy_scores_rows(tmp_path):
    package = make_package(tmp_path, {"Лист1": {"rows": [
        ["q", "pred", "gt"],
        ["в1", "кредит", "кредит"], ["в2", "вклад", "кредит"]]}})
    result = run_package(package, tmp_path / "out", client=FakeClient([
        layout_answer(roles={"query_id": None, "session_id": None,
                              "input_query": "A", "output_answer": "B",
                              "scenario": None, "assessor_id": None,
                              "reference_answers": []}),
        metric_answer(method="accuracy",
                       sources=[{"column_id": "B", "role": "prediction",
                                  "normalization": "label", "polarity": "direct"},
                                 {"column_id": "C", "role": "target",
                                  "normalization": "label", "polarity": "direct"}])]))
    assert sorted(result.umr.frame["main_metric"].tolist()) == [0.0, 1.0]
    assert result.status == "computed"


def test_blank_input_query_rows_are_dropped_not_fatal(tmp_path):
    """Частично пустой input_query во flat-корзине: после исчерпания repair
    строки отбрасываются с аудитом, а не роняют прогон."""
    package = make_package(tmp_path, {"Лист1": {"rows": [
        ["q", "a", "m"], ["в1", "о1", 1], [None, "о2", 1], ["в3", "о3", 0]]}})
    result = run_package(package, tmp_path / "out", client=FakeClient(
        [layout_answer()] * 3 + [metric_answer()]))
    assert result.status == "computed"
    assert len(result.umr.frame) == 2
    assert result.report["dropped_rows"]["blank_input_query"] == [3]


def test_metric_failure_degrades_not_dies(tmp_path):
    package = make_package(tmp_path, {"Лист1": {"rows": [
        ["q", "a", "m"], ["в1", "о1", 1]]}})
    bad = metric_answer(sources=[{"column_id": "ZZ", "role": "final_score",
                                   "normalization": "numeric",
                                   "polarity": "direct"}])
    result = run_package(package, tmp_path / "out",
                          client=FakeClient([layout_answer()] + [bad] * 3))
    assert result.status == "not_computable"
    assert len(result.umr.frame) == 1
    assert result.report["status"] == "not_computable"
    assert result.km["reason_code"] == "not_evaluable"
    assert any(stage["outcome"] == "degraded" for stage in result.report["stages"])


def test_ambiguous_baseline_is_not_repaired_or_moved_to_spare_sheet(tmp_path):
    package = make_package(tmp_path, {
        "Корзина": {"rows": [
            ["q", "a", "m"], ["в1", "о1", 1], ["в2", "о2", 0]]},
        "Справочник": {"rows": [
            ["Сценарий", "Описание"], ["вход", "про вход"]]},
    })
    client = FakeClient([
        layout_answer(sheet_name="Корзина"),
        metric_answer(reported_value={"state": "ambiguous", "value": None, "raw": None}),
    ])

    result = run_package(package, tmp_path / "out", client=client)

    assert result.status == "not_computable"
    assert result.km["reason_code"] == "ambiguous_baseline"
    assert result.report["decisions"]["sheet"] == "Корзина"
    assert client.calls == 2


def test_pinned_sheet_skips_sheet_retry(tmp_path):
    package = make_package(tmp_path, {"Лист1": {"rows": [
        ["q", "a", "m"], ["в1", "о1", 1]]}})
    bad = metric_answer(sources=[{"column_id": "ZZ", "role": "final_score",
                                   "normalization": "numeric",
                                   "polarity": "direct"}])
    client = FakeClient([layout_answer()] + [bad] * 3)
    result = run_package(package, tmp_path / "out", client=client,
                          sheet_name="Лист1")
    assert result.status == "not_computable"
    assert client.labels.count("layout_turn1") == 1

def test_layout_failure_retries_spare_sheet(tmp_path):
    # Первый лист — blob, который не разворачивается вовсе; нода обязана
    # отвергнуть лист и собрать разметку на втором, как делает этап метрики.
    package = make_package(tmp_path, {
        "Диалоги": {"rows": [["dialog", "m"], ["мусор", 1], ["ещё мусор", 0]]},
        "Плоский": {"rows": [["q", "a", "m"], ["в1", "о1", 1], ["в2", "о2", 0]]},
    })
    blob = layout_answer(
        sheet_name="Диалоги",
        roles={"query_id": None, "session_id": None, "input_query": "A",
               "output_answer": None, "scenario": None, "assessor_id": None,
               "reference_answers": []},
        grouping={"kind": "blob_row", "column": None},
        dialogue_blob={"column": "A", "container": "plain_text",
                       "question_marker": "клиент", "answer_marker": "оператор"})
    client = FakeClient([blob, blob, blob,
                         layout_answer(sheet_name="Плоский"), metric_answer()])

    result = run_package(package, tmp_path / "out", client=client)

    assert result.status == "computed"
    assert result.report["decisions"]["sheet"] == "Плоский"



def _blob_layout(container: str, question: str, answer: str):
    return layout_answer(roles={"query_id": None, "session_id": None,
                                "input_query": "A", "output_answer": None,
                                "scenario": None, "assessor_id": None,
                                "reference_answers": []},
                         grouping={"kind": "blob_row", "column": None},
                         dialogue_blob={"column": "A", "container": container,
                                        "question_marker": question,
                                        "answer_marker": answer})


def _score_plan():
    return metric_answer(sources=[{"column_id": "B", "role": "final_score",
                                   "normalization": "numeric", "polarity": "direct"}])


def test_blob_dialogue_role_dicts_expand_turns(tmp_path):
    # Самый частый экспорт чатов: JSON-массив сообщений role/content (LAIM-0073).
    blob = ('[{"role": "user", "content": "привет"}, '
            '{"role": "assistant", "content": "здравствуйте"}, '
            '{"role": "user", "content": "вопрос"}, '
            '{"role": "assistant", "content": "ответ"}]')
    package = make_package(tmp_path, {"Лист1": {"rows": [
        ["messages", "m"], [blob, 1],
        ['[{"role": "user", "content": "ещё"}, {"role": "assistant", "content": "снова"}]', 0]]}})
    result = run_package(package, tmp_path / "out", client=FakeClient([
        _blob_layout("python_list", "user", "assistant"), _score_plan()]))
    turns = ast.literal_eval(result.umr.frame["dialogue"].iloc[0])
    assert [(turn[1], turn[2]) for turn in turns] == [("привет", "здравствуйте"), ("вопрос", "ответ")]
    assert result.status == "computed" and result.report["dropped_rows"] == {}


def test_blob_dialogue_pairs_expand_turns(tmp_path):
    blob = "[('привет', 'здравствуйте'), ('вопрос', 'ответ')]"
    package = make_package(tmp_path, {"Лист1": {"rows": [
        ["dialog", "m"], [blob, 1], ["[('ещё', 'снова')]", 0]]}})
    result = run_package(package, tmp_path / "out", client=FakeClient([
        _blob_layout("python_list", "Q", "A"), _score_plan()]))
    turns = ast.literal_eval(result.umr.frame["dialogue"].iloc[0])
    assert [(turn[1], turn[2]) for turn in turns] == [("привет", "здравствуйте"), ("вопрос", "ответ")]


def test_blob_dialogue_role_dicts_with_russian_roles(tmp_path):
    blob = ('[{"speaker": "Клиент", "text": "привет"}, '
            '{"speaker": "Оператор", "text": "здравствуйте"}]')
    package = make_package(tmp_path, {"Лист1": {"rows": [
        ["dialog", "m"], [blob, 1], [blob, 0]]}})
    result = run_package(package, tmp_path / "out", client=FakeClient([
        _blob_layout("python_list", "Клиент", "Оператор"), _score_plan()]))
    turns = ast.literal_eval(result.umr.frame["dialogue"].iloc[0])
    assert [(turn[1], turn[2]) for turn in turns] == [("привет", "здравствуйте")]


def test_foreign_validation_report_is_flagged(tmp_path):
    # Отчёт о валидации другого агента: baseline берётся из него как есть,
    # но журнал обязан это показать (аудит LAIM-0188, LAIM-0040).
    package = make_package(tmp_path, {"Лист1": {"rows": [
        ["q", "a", "m"], ["в1", "о1", 1], ["в2", "о2", 0]]}},
        validation=("Отчёт о валидации агента CI09000002", "Accuracy равна 0.5"))
    result = run_package(package, tmp_path / "out",
                          client=FakeClient([layout_answer(), metric_answer()]))
    assert result.status == "computed" and result.report["km"]["value"] == 0.5
    warning = next(w for w in result.report["warnings"] if w["code"] == "report_identity_mismatch")
    assert "CI09000002" in warning["message"] and "CI09000001" in warning["message"]


def test_matching_validation_report_is_not_flagged(tmp_path):
    package = make_package(tmp_path, {"Лист1": {"rows": [
        ["q", "a", "m"], ["в1", "о1", 1], ["в2", "о2", 0]]}},
        validation=("Отчёт о валидации агента ci09000001", "Accuracy равна 0.5"))
    result = run_package(package, tmp_path / "out",
                          client=FakeClient([layout_answer(), metric_answer()]))
    assert [w["code"] for w in result.report["warnings"]] == []


def test_percent_point_scores_publish_ratio_with_warning(tmp_path):
    package = make_package(tmp_path, {"Лист1": {"rows": [
        ["q", "a", "m"], ["в1", "о1", 70], ["в2", "о2", 100]]}},
        validation=("Доля верных ответов составила 85%",))
    result = run_package(package, tmp_path / "out", client=FakeClient([
        layout_answer(),
        metric_answer(scale="percent",
                      reported_value={"state": "declared", "value": 85,
                                       "raw": "85%"})]))
    assert result.umr.frame["main_metric"].tolist() == [0.7, 1.0]
    assert result.report["km"]["value"] == 85.0
    assert result.report["km"]["reconciliation"] == "match"
    assert [w["code"] for w in result.report["warnings"]] == ["score_domain_percent"]


def test_spare_sheet_layout_failure_keeps_first_sheet_basket(tmp_path):
    # План метрики не собрался на первом листе, а на запасном (справочнике)
    # модель не дала разметку вовсе: корзина первого листа всё равно
    # публикуется как not_computable — нода не умирает.
    package = make_package(tmp_path, {
        "Корзина": {"rows": [["q", "a", "m"], ["в1", "о1", 1], ["в2", "о2", 0]]},
        "Справочник": {"rows": [["Сценарий", "Описание"], ["вход", "про вход"]]},
    })
    bad = metric_answer(sources=[{"column_id": "ZZ", "role": "final_score",
                                   "normalization": "numeric", "polarity": "direct"}])
    client = FakeClient([layout_answer(sheet_name="Корзина")] + [bad] * 3
                        + ["не могу разметить справочник"] * 3)
    result = run_package(package, tmp_path / "out", client=client)
    assert result.status == "not_computable"
    assert result.umr.frame["input_query"].tolist() == ["в1", "в2"]
    assert result.report["decisions"]["sheet"] == "Корзина"
    assert any(w["code"] == "spec_error" for w in result.report["warnings"])
