"""Структурное чтение DOCX: заголовки, таблицы построчно, абзацы."""
from __future__ import annotations


from helpers import make_docx
from laim_basket.reading.docx_reader import read_document_paragraphs


def test_docx_headings_tables_and_paragraphs(tmp_path):
    path = tmp_path / "report.docx"
    make_docx(
        path,
        heading="Отчет о валидации",
        paragraphs=("Тип валидации: первичная",),
        table=[["Метрика", "Значение", "Порог"], ["Accuracy", "0.93", "0.9"]],
    )
    paragraphs = read_document_paragraphs(path, "document_docx")
    assert paragraphs[0] == "# Отчет о валидации"
    assert "Тип валидации: первичная" in paragraphs
    assert "Метрика | Значение | Порог" in paragraphs
    assert "Accuracy | 0.93 | 0.9" in paragraphs


def test_table_between_paragraphs_keeps_order(tmp_path):
    path = tmp_path / "mixed.docx"
    make_docx(path, paragraphs=("До таблицы",), table=[["a", "b"]])
    paragraphs = read_document_paragraphs(path, "document_docx")
    assert paragraphs.index("До таблицы") < paragraphs.index("a | b")


def test_txt_is_split_to_paragraphs(tmp_path):
    path = tmp_path / "assessor_instruction.txt"
    path.write_text("Правило 1\n\nПравило 2", encoding="utf-8")
    assert read_document_paragraphs(path, "document_txt") == ("Правило 1", "Правило 2")
