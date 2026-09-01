"""Извлечение документов в стабильные span-абзацы средствами stdlib."""

import html
import re
import zipfile
from pathlib import Path

from ..errors import ReadError

_PARAGRAPH_END = re.compile(r"</w:p>")
_CELL_END = re.compile(r"</w:tc>")
_TAG = re.compile(r"<[^>]+>")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def read_docx_text(path: str | Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise ReadError(f"Не удалось прочитать docx: {path}: {exc}", path=str(path)) from exc
    text = _PARAGRAPH_END.sub("\n", xml)
    text = _CELL_END.sub("\t", text)
    text = _TAG.sub("", text)
    return _MANY_NEWLINES.sub("\n\n", html.unescape(text)).strip()


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


def _spans(
    text: str,
    path: str | Path,
    document_id: str,
    format_name: str,
) -> tuple[dict[str, str], ...]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    if not paragraphs:
        raise ReadError(f"{format_name} не содержит текста: {path}", path=str(path))
    return tuple(
        {"id": f"{document_id}:p{index:04d}", "text": text}
        for index, text in enumerate(paragraphs, start=1)
    )


def read_docx_spans(path: str | Path, document_id: str) -> tuple[dict[str, str], ...]:
    return _spans(read_docx_text(path), path, document_id, "DOCX")


def read_txt_spans(path: str | Path, document_id: str) -> tuple[dict[str, str], ...]:
    return _spans(read_txt_text(path), path, document_id, "TXT")


def read_document_spans(
    path: str | Path,
    document_id: str,
    kind: str,
) -> tuple[dict[str, str], ...]:
    readers = {
        "document_docx": read_docx_spans,
        "document_txt": read_txt_spans,
    }
    try:
        reader = readers[kind]
    except KeyError as exc:
        raise ReadError(
            f"Неподдерживаемый формат документа: {kind}",
            path=str(path),
            kind=kind,
        ) from exc
    return reader(path, document_id)
