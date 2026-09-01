"""Структурное чтение документов средствами stdlib: заголовки Word с
'#'-префиксом, строки таблиц через ' | ', нумерация абзацев — в промпте."""

import html
import re
import zipfile
from pathlib import Path

from ..errors import ReadError

_PARAGRAPH_END = re.compile(r"</w:p>")
_TAG = re.compile(r"<[^>]+>")


def read_txt_text(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise ReadError(
            f"TXT должен быть в кодировке UTF-8: {path}: {exc}",
            path=str(path),
        ) from exc
    except OSError as exc:
        raise ReadError(f"Не удалось прочитать TXT: {path}: {exc}", path=str(path)) from exc


_BLOCK = re.compile(r"<w:tbl>.*?</w:tbl>|<w:p[ >/].*?</w:p>|<w:p/>", re.DOTALL)
_ROW = re.compile(r"<w:tr[ >].*?</w:tr>|<w:tr>.*?</w:tr>", re.DOTALL)
_CELL = re.compile(r"<w:tc[ >].*?</w:tc>|<w:tc>.*?</w:tc>", re.DOTALL)
_HEADING = re.compile(r'w:pStyle[^>]+w:val="Heading(\d)"')


def _plain(fragment: str) -> str:
    return html.unescape(_TAG.sub("", _PARAGRAPH_END.sub(" ", fragment))).strip()


def _docx_paragraphs(xml: str) -> tuple[str, ...]:
    """Блоки документа по порядку: таблица -> строки 'a | b', абзац -> текст."""
    lines: list[str] = []
    for block in _BLOCK.finditer(xml):
        fragment = block.group(0)
        if fragment.startswith("<w:tbl>"):
            for row in _ROW.finditer(fragment):
                cells = [_plain(cell.group(0)) for cell in _CELL.finditer(row.group(0))]
                line = " | ".join(cells)
                if line.strip(" |"):
                    lines.append(line)
            continue
        heading = _HEADING.search(fragment)
        text = _plain(fragment)
        if not text:
            continue
        lines.append(f"{'#' * int(heading.group(1))} {text}" if heading else text)
    return tuple(lines)


def read_document_paragraphs(path: str | Path, kind: str) -> tuple[str, ...]:
    """Структурный текст документа: заголовки с '#', таблицы построчно."""
    if kind == "document_txt":
        text = read_txt_text(path)
        paragraphs = tuple(
            part.strip().replace("\n", " ")
            for part in re.split(r"\n\s*\n", text)
            if part.strip()
        )
    elif kind == "document_docx":
        try:
            with zipfile.ZipFile(path) as archive:
                xml = archive.read("word/document.xml").decode("utf-8", errors="replace")
        except (zipfile.BadZipFile, KeyError, OSError) as exc:
            raise ReadError(f"Не удалось прочитать docx: {path}: {exc}", path=str(path)) from exc
        paragraphs = _docx_paragraphs(xml)
    else:
        raise ReadError(f"Неподдерживаемый формат документа: {kind}", path=str(path), kind=kind)
    if not paragraphs:
        raise ReadError(f"Документ не содержит текста: {path}", path=str(path))
    return paragraphs
