"""Чтение входного пакета и проверка его идентичности до вызовов LLM."""

import logging
import re
from pathlib import Path

from .errors import PackageError
from .journal import Journal
from .models import RunContext
from .reading.docx_reader import read_document_paragraphs
from .reading.package_scan import scan_package
from .reading.xlsx_reader import read_workbook

logger = logging.getLogger(__name__)

# Порты документов узнаются по стабильной номенклатуре процесса валидации
# (не пер-корзинное знание): нода получает канонические имена от main.py,
# CLI-пакеты несут человеческие имена тех же трёх типов документов.
_DOCUMENT_PORT_MARKERS = (
    ("validation_report", ("validation_report", "валидац")),
    ("development_report", ("development_report", "разработ")),
    ("assessor_instruction", ("assessor_instruction", "инструкц", "размет")),
)


def _document_ports(names: list[str], kinds: dict[str, str]) -> dict[str, str]:
    ports: dict[str, str] = {}
    for name in names:
        if kinds[name] == "document_txt":
            ports[name] = "assessor_instruction"
    for port, markers in _DOCUMENT_PORT_MARKERS:
        if port in ports.values():
            continue
        matched = [
            name
            for name in names
            if name not in ports and any(marker in name.casefold() for marker in markers)
        ]
        if len(matched) == 1:
            ports[matched[0]] = port
    unassigned = [name for name in names if name not in ports]
    missing = [port for port, _ in _DOCUMENT_PORT_MARKERS if port not in ports.values()]
    if len(unassigned) == 1 and len(missing) == 1:
        ports[unassigned[0]] = missing[0]
    if len(ports) != 3:
        raise PackageError(
            "Не удалось сопоставить документы ролям (валидация/разработка/инструкция)",
            documents=names,
            assigned=ports,
        )
    return ports


_CI_TOKEN = re.compile(r"(?i)\bCI\d{6,}\b")


def check_report_identity(context: RunContext, journal: Journal) -> None:
    """CI-код в отчёте о валидации против basket_id: baseline берётся из отчёта
    как есть, поэтому чужой отчёт обязан быть виден в журнале (LAIM-0188)."""
    text = "\n".join(
        paragraph
        for document in context.documents
        if document["port"] == "validation_report"
        for paragraph in document["paragraphs"]
    )
    tokens = sorted({token.upper() for token in _CI_TOKEN.findall(text)})
    if not tokens:
        logger.info(
            "Отчёт о валидации не содержит CI-кода: идентичность корзины "
            "%s по отчёту не подтверждена",
            context.basket_id,
        )
    elif context.basket_id in tokens:
        logger.info(
            "Идентичность подтверждена: CI %s встречается в отчёте о валидации", context.basket_id
        )
    else:
        journal.warning(
            "report_identity_mismatch",
            f"отчёт о валидации упоминает {tokens}, корзина — {context.basket_id}: "
            "возможен чужой отчёт; baseline взят из него без изменений",
        )


def _basket_id(package_name: str) -> str:
    match = re.search(r"(?i)ci[0-9]+", package_name)
    if match is None:
        logger.warning(
            "CI-код не найден в имени пакета %r — basket_id взят как есть: %r",
            package_name,
            package_name,
        )
        return package_name
    return match.group(0).upper()


def build_run_context(package: str | Path, agent_ci: str = "") -> RunContext:
    identity = (agent_ci or "").strip()
    if identity and not re.fullmatch(r"CI[0-9]+", identity, re.IGNORECASE):
        raise PackageError(
            "Этап L. Ожидалось: agent_ci вида CI и цифры либо пустая строка. "
            f"Получено: {identity!r}. Действие: исправьте настройку agent_ci."
        )
    manifest = scan_package(package)
    baskets = manifest["baskets"]
    documents = manifest["documents"]
    logger.info(
        "Пакет %s: корзин %d %s, документов %d %s",
        manifest["package_name"],
        len(baskets),
        baskets,
        len(documents),
        documents,
    )
    if len(baskets) != 1:
        raise PackageError("Пакет должен содержать ровно одну XLSX-корзину", found=baskets)
    if len(documents) != 3:
        raise PackageError(
            "Пакет должен содержать два DOCX-отчёта и инструкцию (DOCX или UTF-8 TXT)",
            found=documents,
        )
    files_by_name = {item["name"]: item for item in manifest["files"]}
    kinds = {name: files_by_name[name]["kind"] for name in documents}
    ports = _document_ports(sorted(documents), kinds)
    package_dir = Path(manifest["package_dir"])
    order = {port: index for index, (port, _) in enumerate(_DOCUMENT_PORT_MARKERS)}
    loaded_documents = tuple(
        sorted(
            (
                {
                    "port": ports[name],
                    "name": name,
                    "paragraphs": read_document_paragraphs(package_dir / name, kinds[name]),
                }
                for name in documents
            ),
            key=lambda document: order[document["port"]],
        )
    )
    basket_path = package_dir / baskets[0]
    sheets = read_workbook(basket_path)
    logger.info("Книга %s: листы %s", basket_path.name, list(sheets))
    return RunContext(
        basket_id=identity.upper() if identity else _basket_id(manifest["package_name"]),
        file_hashes={item["name"]: item["sha256"] for item in manifest["files"]},
        sheets=sheets,
        documents=loaded_documents,
    )
