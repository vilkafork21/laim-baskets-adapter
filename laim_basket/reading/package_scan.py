"""Поиск одной книги и трёх документов без опоры на семантику имён файлов."""

import hashlib
import logging
import zipfile
from pathlib import Path

from ..errors import PackageError


logger = logging.getLogger(__name__)

_TXT_INSTRUCTION_NAME = "assessor_instruction.txt"


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _ooxml_kind(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile:
        return "other"
    if any(name.startswith("xl/") for name in names):
        return "basket_xlsx"
    if any(name.startswith("word/") for name in names):
        return "document_docx"
    return "other"


def classify_file(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            if path.suffix.casefold() == ".txt":
                return "document_txt"
            if handle.read(2) != b"PK":
                return "other"
    except OSError as exc:
        raise PackageError(f"Не удалось прочитать файл {path}: {exc}", path=str(path)) from exc
    return _ooxml_kind(path)


def scan_package(input_path: str | Path) -> dict[str, object]:
    root = Path(input_path)
    if not root.exists():
        raise PackageError(f"Путь не существует: {root}", path=str(root))
    paths = [root] if root.is_file() else sorted(
        path for path in root.iterdir()
        if path.is_file() and not path.name.startswith((".", "~$"))
    )
    files = []
    baskets: list[str] = []
    documents: list[str] = []
    for path in paths:
        kind = classify_file(path)
        logger.debug("Файл %s классифицирован как %s (%d байт)", path.name, kind, path.stat().st_size)
        files.append({
            "name": path.name,
            "kind": kind,
            "sha256": _sha256(path),
        })
        if kind == "basket_xlsx":
            baskets.append(path.name)
        elif kind == "document_docx":
            documents.append(path.name)
        elif kind == "document_txt" and path.name.casefold() == _TXT_INSTRUCTION_NAME:
            documents.append(path.name)
    package_dir = root if root.is_dir() else root.parent
    return {
        "package_dir": str(package_dir),
        "package_name": package_dir.name,
        "files": files,
        "baskets": baskets,
        "documents": documents,
    }
