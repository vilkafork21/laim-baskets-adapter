"""Снимок книги: якорные строки, статистика колонок, ограниченный размер."""
from __future__ import annotations

import json


from helpers import make_workbook
from laim_basket.evidence.workbook import workbook_evidence
from laim_basket.reading.xlsx_reader import read_workbook


def _sheets(tmp_path, rows, merges=()):
    path = tmp_path / "book.xlsx"
    make_workbook(path, {"Лист1": {"rows": rows, "merges": list(merges)}})
    return read_workbook(path)


def test_snapshot_has_anchors_and_column_stats(tmp_path):
    rows = [["q", "a", "mark"]] + [[f"вопрос {i}", f"ответ {i}", 1] for i in range(200)]
    snapshot = workbook_evidence(_sheets(tmp_path, rows))
    sheet = snapshot["sheets"][0]
    assert sheet["name"] == "Лист1"
    assert sheet["header_candidate_row"] == 1
    assert sheet["anchor_rows"][0]["row"] == 1
    addresses = [column["address"] for column in sheet["columns"]]
    assert addresses[:3] == ["A", "B", "C"]
    mark = sheet["columns"][2]
    assert mark["non_null"] == 200 and mark["numeric"] == 200


def test_snapshot_carries_merges_and_dialogue_hint(tmp_path):
    rows = [["s", "messages", "m"],
            [1, "КЛИЕНТ: привет АГЕНТ: здравствуйте КЛИЕНТ: вопрос", 1],
            [2, "КЛИЕНТ: ещё АГЕНТ: ответ АГЕНТ: уточнение", 0]]
    snapshot = workbook_evidence(_sheets(tmp_path, rows, merges=["A2:A3"]))
    sheet = snapshot["sheets"][0]
    assert "A2:A3" in sheet["merged"]
    assert sheet["dialogue_markers"] == ["КЛИЕНТ", "АГЕНТ"]


def test_snapshot_size_does_not_grow_with_rows(tmp_path):
    small = json.dumps(workbook_evidence(_sheets(tmp_path, [["q", "m"]] + [["в", 1]] * 50)))
    big = json.dumps(workbook_evidence(_sheets(tmp_path, [["q", "m"]] + [["в", 1]] * 5000)))
    assert len(big) < len(small) * 3
