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


def test_attributed_table_alternate_prefix_nested_cell_and_fields(tmp_path):
    import zipfile
    path = tmp_path / 'namespaced.docx'
    xml = '''<x:document xmlns:x="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <x:body><x:p><x:pPr><x:pStyle x:val="2"/></x:pPr><x:r><x:t>До</x:t></x:r></x:p>
    <x:tbl x:rsidR="01"><x:tr x:rsidR="02"><x:tc><x:p><x:r><x:t>Accuracy</x:t></x:r></x:p></x:tc>
    <x:tc><x:p><x:r><x:instrText>HYPERLINK мусор</x:instrText><x:t>0.93</x:t></x:r></x:p>
    <x:tbl><x:tr><x:tc><x:p><x:r><x:t>nested</x:t></x:r></x:p></x:tc></x:tr></x:tbl></x:tc>
    </x:tr></x:tbl><x:p><x:r><x:t>После</x:t></x:r></x:p></x:body></x:document>'''
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('word/document.xml', xml)
    assert read_document_paragraphs(path, 'document_docx') == ('## До', 'Accuracy | 0.93 nested', 'После')


def test_malformed_docx_is_structured_read_error(tmp_path):
    import zipfile
    import pytest
    from laim_basket.errors import ReadError
    path = tmp_path / 'broken.docx'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('word/document.xml', '<w:document>')
    with pytest.raises(ReadError):
        read_document_paragraphs(path, 'document_docx')
