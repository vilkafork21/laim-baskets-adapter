"""Промпты: документы в XML-тегах ролей, схема в конце, контекст перед задачей."""
from __future__ import annotations


from laim_basket.llm.prompts import (
    layout_messages,
    metric_messages,
)

DOCS = (
    {"port": "validation_report", "name": "validation_report.docx",
     "paragraphs": ("# Отчет о валидации", "Accuracy | 0.93 | 0.9")},
    {"port": "development_report", "name": "development_report.docx",
     "paragraphs": ("Расшифровка колонок: mark - оценка",)},
    {"port": "assessor_instruction", "name": "assessor_instruction.txt",
     "paragraphs": ("Оценка 1/0",)},
)


def test_layout_prompt_wraps_documents_and_ends_with_schema():
    messages = layout_messages({"sheets": []}, DOCS, "", frozenset())
    user = messages[-1]["content"]
    assert "<validation_report>" in user and "</validation_report>" in user
    assert "<development_report>" in user and "<assessor_instruction>" in user
    assert "Расшифровка колонок: mark - оценка" in user
    assert user.index("<validation_report>") < user.index("JSON SCHEMA")
    assert "json_schema" not in messages[0]["content"]
    assert "assessor_id — только идентификатор" in messages[0]["content"]


def test_layout_prompt_names_pinned_and_rejected_sheets():
    messages = layout_messages({"sheets": []}, DOCS, "Лист2", frozenset({"Лист1"}))
    user = messages[-1]["content"]
    assert "Лист2" in user and "Лист1" in user


def test_metric_prompt_carries_inventory_and_priority_rule():
    inventory = [{"column_id": "C", "header": "mark", "non_null": 2, "samples": [1, 0]}]
    messages = metric_messages(inventory, DOCS, {"grouping": "none", "unit": "turn"})
    user = messages[-1]["content"]
    assert "mark" in user
    assert "отчёт о валидации" in messages[0]["content"].lower()
    assert "identity требует ровно один final_score" in messages[0]["content"]


def test_truncated_document_is_logged(monkeypatch, caplog):
    import logging

    from laim_basket import defaults

    monkeypatch.setattr(defaults, "DOCUMENT_CHAR_CAP", 40)
    long_docs = ({"port": "validation_report", "name": "validation_report.docx",
                  "paragraphs": tuple(f"абзац {index} " * 3 for index in range(10))},)
    with caplog.at_level(logging.WARNING):
        messages = layout_messages({"sheets": []}, long_docs, "", frozenset())
    assert "validation_report" in caplog.text and "обрезан" in caplog.text
    assert "абзац 9" not in messages[-1]["content"]


def test_short_document_is_not_logged(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        layout_messages({"sheets": []}, DOCS, "", frozenset())
    assert "обрезан" not in caplog.text
