"""Сквозной прогон ноды: неизвестная метрика в отчёте → not_computable, не цвет.

Реальный run_package (xlsx + три docx, layout, план, движок, гейт), LLM
подменён детерминированными ответами. Сценарий «новый агент»: отчёт о валидации
называет F1 = 0.82, а единственная построчная колонка корзины даёт среднее 0.75.
Ни один план реестра не воспроизводит 0.82 — нода обязана отказаться с кодом
km_reconciliation_mismatch. Контрольный случай: отчёт называет 0.75 — computed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from docx import Document
from openpyxl import Workbook

import main as node
from laim_basket.pipeline import run_package

BASKET_ID = "CI00000001"


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    document = Document()
    for text in paragraphs:
        document.add_paragraph(text)
    document.save(path)


def _write_package(root: Path, reported: str, scores: list[int]) -> Path:
    package = root / BASKET_ID
    package.mkdir()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Лист1"
    sheet.append(["query_id", "input_query", "output_answer", "Итог"])
    for index, score in enumerate(scores, start=1):
        sheet.append([f"q{index}", f"вопрос {index}", f"ответ {index}", score])
    workbook.save(package / "test_set.xlsx")
    _write_docx(package / "assessor_instruction.docx", [
        "Инструкция ассессора: каждая пара запрос-ответ оценивается независимо.",
        "Итог равен 1, если ответ верный, иначе 0.",
    ])
    _write_docx(package / "development_report.docx", [
        "Отчёт о разработке агента.",
        "Ключевая метрика качества описана в отчёте о валидации.",
    ])
    _write_docx(package / "validation_report.docx", [
        "Отчёт о валидации агента.",
        "Ключевая метрика: F1 на тестовой корзине.",
        f"Значение метрики: {reported}",
    ])
    return package


LAYOUT = {
    "layout_version": "laim-layout.v1",
    "basket_id": BASKET_ID,
    "sheet_name": "Лист1",
    "ignored_sheets": [],
    "header_rows": [1],
    "roles": {
        "input_query": {"column": "B", "header": "input_query"},
        "output_answer": {"column": "C", "header": "output_answer"},
        "query_id": {"column": "A", "header": "query_id"},
        "scenario": None,
        "assessor_id": None,
        "reference_answers": [],
    },
    "grouping": {"kind": "none", "column": None},
    "dialogue_blob": None,
    "weight": None,
}


def _measurement(reported: str) -> dict:
    return {
        "plan_version": "laim-measurement-plan.v2",
        "basket_id": BASKET_ID,
        "metric_name": "F1",
        "document_roles": {
            "instruction": "doc-1",
            "development_report": "doc-2",
            "validation_report": "doc-3",
        },
        "assessment_mode": "qa",
        "score": {
            "method": "identity",
            "sources": [{
                "column_id": "D", "role": "final_score",
                "normalization": "numeric", "polarity": "direct",
            }],
            "missing_policy": "fail",
            "majority_denominator": None,
        },
        "reducer": {"method": "mean"},
        "release": {"threshold": None, "comparator": None, "scale": "ratio", "precision": 2},
        "reported_value_state": "unambiguous",
        "reported_value": {"value": reported, "raw": reported, "span_id": "doc-3:p0003"},
        "evidence": {
            "metric": ["doc-3:p0002"],
            "score": ["doc-1:p0002"],
            "assessment_mode": ["doc-1:p0001"],
            "missing_policy": [],
            "denominator": [],
            "reducer": ["doc-3:p0002"],
            "release": [],
            "reported_value": ["doc-3:p0003"],
        },
    }


class ScriptedLlm:
    """Возвращает один и тот же layout и план: модели больше нечего предложить."""

    def __init__(self, reported: str):
        self.reported = reported
        self.calls: list[str] = []

    def chat(self, messages: list[dict], label: str) -> str:
        self.calls.append(label)
        proposal = LAYOUT if label.startswith("layout") else _measurement(self.reported)
        return json.dumps(proposal, ensure_ascii=False)


def _run(tmp_path: Path, reported: str, scores: list[int]):
    package = _write_package(tmp_path, reported, scores)
    client = ScriptedLlm(reported)
    result = run_package(package, tmp_path / "out", client=client)
    return result, client


def test_metric_outside_registry_is_refused_with_reason(tmp_path):
    result, client = _run(tmp_path, reported="0.82", scores=[1, 1, 1, 0])
    assert result.status == "not_evaluable"
    assert result.km["reason_code"] == "km_reconciliation_mismatch"
    assert result.km["details"]["metric_name"] == "F1"
    assert result.km["details"]["recomputed_value"] == pytest.approx(0.75)
    # модель получила repair-попытки и всё равно не смогла воспроизвести отчёт
    assert sum(label.startswith("measurement") for label in client.calls) == 5

    contract = node._monitoring_metric(result)
    assert contract["status"] == "not_computable"
    assert contract["reason_code"] == "km_reconciliation_mismatch"
    assert "baseline" not in contract or contract["baseline"].get("value") is None


def test_reproduced_report_value_is_published(tmp_path):
    result, _client = _run(tmp_path, reported="0.75", scores=[1, 1, 1, 0])
    assert result.status == "computed"
    assert result.km["reconciliation"]["status"] == "match"
    contract = node._monitoring_metric(result)
    assert contract["status"] == "computed"
    assert contract["baseline"]["value"] == pytest.approx(0.75)
    assert contract["baseline"]["reconciliation"] == "match"
    assert contract["scoring"]["sources"][0]["column_name"] == "итог_metric"
