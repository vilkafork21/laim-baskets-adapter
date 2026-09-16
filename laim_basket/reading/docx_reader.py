"""Структурное чтение Word XML: блоки body, строки таблиц и заголовки."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from ..errors import ReadError

_WORD_NAMESPACES = {
    "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "http://purl.oclc.org/ooxml/wordprocessingml/main",
}
_HEADING = re.compile(r"(?:Heading)?([1-9])", re.IGNORECASE)


def read_txt_text(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8-sig").strip()
    except (UnicodeDecodeError, OSError) as exc:
        raise ReadError(f"Не удалось прочитать UTF-8 TXT: {path}: {exc}", path=str(path)) from exc


def _text(node: ET.Element, ns: str) -> str:
    parts = []

    def visit(element):
        if element.tag in {ns + "del", ns + "moveFrom", ns + "drawing", ns + "pict"}:
            return
        if element.tag == ns + "t":
            parts.append(element.text or "")
        elif element.tag in {ns + "tab", ns + "br", ns + "cr"}:
            parts.append(" ")
        else:
            for child in element:
                visit(child)

    visit(node)
    return "".join(parts).strip()


def _children(node: ET.Element, tag: str, ns: str):
    """Прямые элементы, включая содержимое Word content controls."""
    for child in node:
        if child.tag == ns + tag:
            yield child
        elif child.tag in {ns + "sdt", ns + "sdtContent", ns + "customXml"}:
            yield from _children(child, tag, ns)


def _blocks(node: ET.Element, ns: str, *, headings: bool = True):
    for child in node:
        if child.tag == ns + "p":
            text = _text(child, ns)
            if not text:
                continue
            style = child.find(f"{ns}pPr/{ns}pStyle")
            heading = _HEADING.fullmatch(style.get(ns + "val", "")) if style is not None else None
            yield f"{'#' * int(heading.group(1))} {text}" if headings and heading else text
        elif child.tag == ns + "tbl":
            for row in _children(child, "tr", ns):
                cells = [
                    " ".join(_blocks(cell, ns, headings=False)) for cell in _children(row, "tc", ns)
                ]
                line = " | ".join(cells)
                if line.strip(" |"):
                    yield line
        elif child.tag in {ns + "sdt", ns + "sdtContent", ns + "customXml"}:
            yield from _blocks(child, ns, headings=headings)


def _docx_paragraphs(xml: bytes | str) -> tuple[str, ...]:
    # Word XML не нуждается в DTD. Запрещаем расширение пользовательских сущностей.
    upper = xml.upper()
    forbidden = (b"<!DOCTYPE", b"<!ENTITY") if isinstance(xml, bytes) else ("<!DOCTYPE", "<!ENTITY")
    if any(token in upper for token in forbidden):
        raise ValueError("DTD и ENTITY в документе Word не поддерживаются")
    root = ET.fromstring(xml)
    namespace = root.tag[1:].split("}", 1)[0] if root.tag.startswith("{") else ""
    if namespace not in _WORD_NAMESPACES:
        raise ValueError("Неизвестное пространство имён WordprocessingML")
    ns = "{" + namespace + "}"
    body = root.find(ns + "body")
    if body is None:
        raise ValueError("В документе отсутствует body")
    return tuple(_blocks(body, ns))


def read_document_paragraphs(path: str | Path, kind: str) -> tuple[str, ...]:
    if kind == "document_txt":
        text = read_txt_text(path)
        paragraphs = tuple(
            part.strip().replace("\n", " ") for part in re.split(r"\n\s*\n", text) if part.strip()
        )
    elif kind == "document_docx":
        try:
            with zipfile.ZipFile(path) as archive:
                paragraphs = _docx_paragraphs(archive.read("word/document.xml"))
        except (zipfile.BadZipFile, KeyError, OSError, ET.ParseError, ValueError) as exc:
            raise ReadError(
                f"Не удалось прочитать DOCX: {path}: {exc}. "
                "Действие: пересохраните документ как корректный DOCX.",
                path=str(path),
            ) from exc
    else:
        raise ReadError(f"Неподдерживаемый формат документа: {kind}", path=str(path), kind=kind)
    if not paragraphs:
        raise ReadError(f"Документ не содержит текста: {path}", path=str(path))
    return paragraphs
