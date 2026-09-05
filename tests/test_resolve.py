"""Физика разметки: границы, роли, группировка — без доказательного слоя."""
from __future__ import annotations


import pytest


from conftest import layout_answer
from helpers import make_workbook
from laim_basket.errors import LayoutError
from laim_basket.reading.xlsx_reader import read_workbook
from laim_basket.resolve import resolve_layout

BASKET = "CI09000001"


def _sheets(tmp_path, rows, merges=()):
    path = tmp_path / "b.xlsx"
    make_workbook(path, {"Лист1": {"rows": rows, "merges": list(merges)}})
    return read_workbook(path)


def test_bounds_stop_at_last_nonempty_input_query(tmp_path):
    rows = [["q", "a", "m"], ["в1", "о1", 1], ["в2", "о2", 0],
            [None, None, None], [None, "итого", 0.5]]
    layout = resolve_layout(layout_answer(), _sheets(tmp_path, rows),
                            BASKET, "", frozenset())
    assert layout.first_data_row == 2 and layout.last_data_row == 3


def test_unknown_sheet_is_layout_error(tmp_path):
    with pytest.raises(LayoutError):
        resolve_layout(layout_answer(sheet_name="Нет такого"),
                       _sheets(tmp_path, [["q", "a", "m"], ["в", "о", 1]]),
                       BASKET, "", frozenset())


def test_pinned_sheet_overrides_model_choice(tmp_path):
    with pytest.raises(LayoutError):
        resolve_layout(layout_answer(),
                       _sheets(tmp_path, [["q", "a", "m"], ["в", "о", 1]]),
                       BASKET, "Другой", frozenset())


def test_rejected_sheet_is_not_reused(tmp_path):
    with pytest.raises(LayoutError):
        resolve_layout(layout_answer(),
                       _sheets(tmp_path, [["q", "a", "m"], ["в", "о", 1]]),
                       BASKET, "", frozenset({"Лист1"}))


def test_one_column_cannot_carry_two_roles(tmp_path):
    with pytest.raises(LayoutError):
        resolve_layout(
            layout_answer(roles={"query_id": None, "session_id": None,
                              "input_query": "A", "output_answer": "B",
                              "scenario": "A", "assessor_id": None,
                              "reference_answers": []}),
            _sheets(tmp_path, [["q", "a", "m"], ["в", "о", 1]]),
            BASKET, "", frozenset())


def test_vertical_merge_forces_merged_rows(tmp_path):
    rows = [["s", "q", "a", "m"], [1, "в1", "о1", 1], [None, "в2", "о2", None]]
    layout = resolve_layout(
        layout_answer(roles={"query_id": None, "session_id": "A",
                          "input_query": "B", "output_answer": "C",
                          "scenario": None, "assessor_id": None,
                          "reference_answers": []}),
        _sheets(tmp_path, rows, merges=["A2:A3", "D2:D3"]),
        BASKET, "", frozenset())
    assert layout.grouping["kind"] == "merged_rows"


def test_non_integer_weight_is_layout_error(tmp_path):
    rows = [["q", "a", "w", "m"], ["в", "о", 1.5, 1]]
    with pytest.raises(LayoutError):
        resolve_layout(layout_answer(weight_column="C"),
                       _sheets(tmp_path, rows), BASKET, "", frozenset())

def test_unusable_session_source_degrades_to_group_index(tmp_path):
    # Пропуски в session-колонке — свойство корзины (обрыв заливки, не merge):
    # нода деградирует к номеру группы, как query_id к синтезу, а не падает.
    from laim_basket.transform.canon import build_canon
    from laim_basket.transform.grouping import apply_grouping

    rows = [["s", "q", "a", "m"],
            ["d1", "в1", "о1", 1],
            [None, "в2", "о2", 0],
            ["d2", "в3", "о3", 1]]
    sheets = _sheets(tmp_path, rows)
    layout = resolve_layout(
        layout_answer(roles={"query_id": None, "session_id": "A",
                          "input_query": "B", "output_answer": "C",
                          "scenario": None, "assessor_id": None,
                          "reference_answers": []},
                      grouping={"kind": "column", "column": "A"}),
        sheets, BASKET, "", frozenset())
    grouped = apply_grouping(sheets["Лист1"], layout.region, layout.transform_config())
    frame, conversion = build_canon(grouped, layout.region, layout.transform_config())

    assert frame["session_id"].tolist() == [0, 0, 1]
    identity = conversion["identity"]["session_id"]
    assert identity["source_rejected"] == "s"
    assert "пропуски" in identity["reason"]
    assert conversion["umr_validation"]["status"] == "passed"



def test_footer_label_in_query_column_is_not_data(tmp_path):
    # Метка «ИТОГО» под запросами при SUBTOTAL/СУММ в колонке оценки — футер
    # таблицы, а не единица оценки (аудит LAIM-0191).
    rows = [["q", "a", "m"], ["в1", "о1", 1], ["в2", "о2", 0],
            ["ИТОГО", None, "=SUBTOTAL(9,C2:C3)"]]
    layout = resolve_layout(layout_answer(), _sheets(tmp_path, rows),
                            BASKET, "", frozenset())
    assert layout.last_data_row == 3


def test_row_formula_row_stays_data(tmp_path):
    rows = [["q", "a", "m"], ["в1", "о1", 1], ["в2", "о2", "=IF(A3=\"в2\",1,0)"]]
    layout = resolve_layout(layout_answer(), _sheets(tmp_path, rows),
                            BASKET, "", frozenset())
    assert layout.last_data_row == 3


def test_numeric_reference_is_not_confused_with_a_score_column(tmp_path):
    rows = [["q", "a", "Эталон числа"], ["сколько", "42", 42]]
    answer = layout_answer(roles={"query_id": None, "session_id": None,
                                  "input_query": "A", "output_answer": "B",
                                  "scenario": None, "assessor_id": None,
                                  "reference_answers": ["C"]})
    layout = resolve_layout(answer, _sheets(tmp_path, rows), BASKET, "", frozenset())
    assert layout.roles["reference_answers"] == ["Эталон числа"]


def test_text_reference_answer_is_accepted(tmp_path):
    rows = [["q", "a", "ref"], ["в1", "о1", "эталон 1"], ["в2", "о2", None]]
    answer = layout_answer(roles={"query_id": None, "session_id": None,
                                  "input_query": "A", "output_answer": "B",
                                  "scenario": None, "assessor_id": None,
                                  "reference_answers": ["C"]})
    layout = resolve_layout(answer, _sheets(tmp_path, rows), BASKET, "", frozenset())
    assert layout.roles["reference_answers"] == ["ref"]


def test_binary_column_cannot_be_assessor_id(tmp_path):
    # Колонка голосов 0/1, названная assessor_id, блокирует план метрики без
    # шанса на repair — как и числовой эталон.
    rows = [["q", "a", "Разметчик 1", "Разметчик 2"], ["в1", "о1", 1, 0], ["в2", "о2", 0, 1]]
    answer = layout_answer(roles={"query_id": None, "session_id": None,
                                  "input_query": "A", "output_answer": "B",
                                  "scenario": None, "assessor_id": "C",
                                  "reference_answers": []})
    with pytest.raises(LayoutError, match="0/1"):
        resolve_layout(answer, _sheets(tmp_path, rows), BASKET, "", frozenset())


def test_numeric_assessor_id_is_accepted(tmp_path):
    rows = [["q", "a", "ассессор"], ["в1", "о1", 17], ["в2", "о2", 42]]
    answer = layout_answer(roles={"query_id": None, "session_id": None,
                                  "input_query": "A", "output_answer": "B",
                                  "scenario": None, "assessor_id": "C",
                                  "reference_answers": []})
    layout = resolve_layout(answer, _sheets(tmp_path, rows), BASKET, "", frozenset())
    assert layout.roles["assessor_id"] == {"source": "ассессор"}
