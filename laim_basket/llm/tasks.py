"""Оркестрация двух LLM-задач: разметка книги и план метрики.

Разметка обязана собрать обязательные поля спеки (иначе SpecError — нода
падает); план метрики деградирует: его провал оставляет корзину публикуемой.
КМ публикуется только заявленной в отчёте о валидации: цитата значения
проверяется кодом, отсутствие значения в отчёте — дефект артефактов.
"""
from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..errors import (
    LayoutError,
    PackageError,
    SpecError,
    StructuredOutputError,
)
from ..evidence.workbook import workbook_evidence
from ..journal import Journal
from ..metric.engine import evaluate
from ..metric.resolve import (
    resolve_measurement_plan,
    verify_reported_citation,
    vertically_merged_source,
)
from ..models import MeasurementPlan, ResolvedLayout, RunContext
from ..publish import PublishedUmr, publish_umr
from ..reading.docx_reader import read_document_paragraphs
from ..reading.package_scan import scan_package
from ..reading.xlsx_reader import read_workbook
from ..resolve import resolve_layout
from ..transform.canon import build_canon
from ..transform.grouping import apply_grouping
from ..transform.values import blank
from .client import request_structured
from .prompts import layout_messages, metric_messages
from .schemas import LAYOUT_SCHEMA, METRIC_SCHEMA

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
            name for name in names
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
            documents=names, assigned=ports,
        )
    return ports


_CI_TOKEN = re.compile(r"(?i)\bCI\d{6,}\b")


def check_report_identity(context: RunContext, journal: Journal) -> None:
    """CI-код в отчёте о валидации против basket_id: baseline берётся из отчёта
    как есть, поэтому чужой отчёт обязан быть виден в журнале (LAIM-0188)."""
    text = "\n".join(
        paragraph
        for document in context.documents if document["port"] == "validation_report"
        for paragraph in document["paragraphs"]
    )
    tokens = sorted({token.upper() for token in _CI_TOKEN.findall(text)})
    if not tokens:
        logger.info("Отчёт о валидации не содержит CI-кода: идентичность корзины "
                    "%s по отчёту не подтверждена", context.basket_id)
    elif context.basket_id in tokens:
        logger.info("Идентичность подтверждена: CI %s встречается в отчёте о валидации",
                    context.basket_id)
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
            "CI-код не найден в имени пакета %r — basket_id взят как есть: %r. "
            "Потребители контура сверяют его с run_context.agent_ci",
            package_name, package_name,
        )
        return package_name
    return match.group(0).upper()


def build_run_context(package: str | Path) -> RunContext:
    manifest = scan_package(package)
    baskets = manifest["baskets"]
    documents = manifest["documents"]
    logger.info(
        "Пакет %s: корзин %d %s, документов %d %s",
        manifest["package_name"], len(baskets), baskets, len(documents), documents,
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
    loaded_documents = tuple(sorted(
        (
            {
                "port": ports[name],
                "name": name,
                "paragraphs": read_document_paragraphs(package_dir / name, kinds[name]),
            }
            for name in documents
        ),
        key=lambda document: order[document["port"]],
    ))
    basket_path = package_dir / baskets[0]
    sheets = read_workbook(basket_path)
    logger.info("Книга %s: листы %s", basket_path.name, list(sheets))
    return RunContext(
        basket_id=_basket_id(manifest["package_name"]),
        file_hashes={item["name"]: item["sha256"] for item in manifest["files"]},
        sheets=sheets,
        documents=loaded_documents,
    )


@dataclass(frozen=True)
class LayoutOutcome:
    proposal: dict
    layout: ResolvedLayout
    frame: pd.DataFrame
    conversion: dict


def _materialize(proposal: dict, ctx: RunContext, pinned_sheet: str,
                 rejected_sheets: frozenset[str]):
    layout = resolve_layout(proposal, ctx.sheets, ctx.basket_id,
                            pinned_sheet, rejected_sheets)
    sheet = ctx.sheets[layout.sheet_name]
    grouped = apply_grouping(sheet, layout.region, layout.transform_config())
    frame, conversion = build_canon(grouped, layout.region, layout.transform_config())
    return layout, frame, conversion


def run_layout(client, ctx: RunContext, journal: Journal, pinned_sheet: str,
               rejected_sheets: frozenset[str]) -> LayoutOutcome:
    evidence = workbook_evidence(ctx.sheets)
    resolved: dict = {}
    recoverable: dict = {}
    attempted: dict = {}

    def validate(proposal: dict) -> None:
        attempted["sheet"] = proposal.get("sheet_name")
        layout, frame, conversion = _materialize(proposal, ctx, pinned_sheet, rejected_sheets)
        validation = conversion["umr_validation"]
        undecodable = conversion["grouping"].get("undecodable_blob_rows")
        if undecodable and validation["status"] == "passed":
            # Строгость остаётся в repair-цикле: модель может исправить маркеры
            # или контейнер; если разметка верна — строки отбросит фолбэк.
            error = LayoutError(
                "Dialogue blob содержит непустые строки, которые нельзя полностью развернуть",
                invalid_rows=conversion["grouping"]["undecodable_blob_samples"],
                invalid_count=len(undecodable),
                repair_hint=(
                    "Проверь question/answer-маркеры и контейнер blob; если "
                    "разметка верна — неразворачиваемые строки будут отброшены."
                ),
            )
            recoverable.update(
                error=error, proposal=copy.deepcopy(proposal),
                layout=layout, frame=frame, conversion=conversion,
                dropped={"undecodable_dialogue_blob": list(undecodable)},
            )
            raise error
        if validation["status"] != "passed":
            error = LayoutError(
                "Канонический UMR не прошёл валидацию",
                **validation,
                repair_hint=(
                    "Смотри source_rows нарушений: пустой input_query внутри "
                    "диапазона данных означает неверно выбранную колонку или "
                    "неучтённую структуру (merged_rows/blob_row)."
                ),
            )
            missing_values = validation["missing_required_values"]
            # Построчные дефекты (пустой input_query, строка вне группы)
            # устранимы отбросом строк после исчерпания repair — фиксируем
            # кандидата на фолбэк. Структурные нарушения чинит только модель.
            blank_query = set(
                missing_values.get("input_query", {}).get("row_positions", []))
            blank_group = {
                position
                for violation in validation["context_violations"]
                if violation["reason"] == "blank_group"
                for position in violation["row_positions"]
            }
            row_scoped = (
                not validation["missing_required"]
                and not validation["type_violations"]
                and set(missing_values) <= {"input_query"}
                and all(violation["reason"] == "blank_group"
                        for violation in validation["context_violations"])
            )
            bad = blank_query | blank_group
            if row_scoped and bad and len(bad) < len(frame):
                keep = [position not in bad for position in range(len(frame))]
                kept = frame.loc[keep].reset_index(drop=True)
                dropped = {
                    kind: sorted(int(frame["source_row_id"].iloc[position])
                                 for position in positions)
                    for kind, positions in (("blank_input_query", blank_query),
                                            ("blank_group", blank_group - blank_query))
                    if positions
                }
                fixed = copy.deepcopy(conversion)
                fixed["row_accounting"]["canon_rows"] = len(kept)
                fixed["row_accounting"]["dropped_invalid_rows"] = sorted(
                    row for rows in dropped.values() for row in rows)
                fixed["umr_validation"]["status"] = "passed"
                fixed["umr_validation"]["missing_required_values"] = {}
                fixed["umr_validation"]["context_violations"] = []
                recoverable.update(
                    error=error, proposal=copy.deepcopy(proposal),
                    layout=layout, frame=kept, conversion=fixed,
                    dropped=dropped,
                )
            raise error
        resolved.update(proposal=proposal, layout=layout, frame=frame,
                         conversion=conversion)

    try:
        request_structured(
            client,
            layout_messages(evidence, ctx.documents, pinned_sheet, rejected_sheets),
            LAYOUT_SCHEMA, "layout", validate_extra=validate,
        )
    except (LayoutError, StructuredOutputError) as exc:
        if recoverable.get("error") is not exc:
            details = dict(exc.details)
            if attempted.get("sheet"):
                # Лист последней попытки нужен pipeline для запасного листа.
                details["sheet"] = attempted["sheet"]
            raise SpecError(
                f"Обязательные поля спеки не собраны после repair: {exc}",
                **details,
            ) from exc
        # Repair не помог, но кандидат публикуем: отброс меньшинства строк —
        # деградация с учётом, а не падение ноды.
        for kind, rows in recoverable["dropped"].items():
            journal.dropped(kind, rows)
        resolved.update(proposal=recoverable["proposal"],
                         layout=recoverable["layout"],
                         frame=recoverable["frame"],
                         conversion=recoverable["conversion"])
    for field in ("session_id", "query_id"):
        entry = resolved["conversion"]["identity"].get(field, {})
        if "source_rejected" in entry:
            journal.warning(
                f"{field}_source_rejected",
                f"колонка {entry['source_rejected']!r} непригодна "
                f"({entry['reason']}) — {field} выведен автоматически",
            )
    return LayoutOutcome(**resolved)


def _column_inventory(sheet, frame, layout: ResolvedLayout) -> list[dict]:
    result = []
    for column_id, name in layout.column_names.items():
        values = frame[name].tolist()
        present = [value for value in values if not blank(value)]
        result.append({
            "column_id": column_id,
            "canonical_raw_name": name,
            "non_null": len(present),
            "samples": [str(value)[:120] for value in present[:3]],
            "formula_rows": list(layout.formula_rows.get(column_id, ())),
            "vertically_merged": vertically_merged_source(sheet, layout, column_id),
        })
    return result


def run_metric(client, ctx: RunContext, outcome: LayoutOutcome,
               journal: Journal) -> tuple[MeasurementPlan, dict, PublishedUmr]:
    layout, frame = outcome.layout, outcome.frame
    sheet = ctx.sheets[layout.sheet_name]
    resolved: dict = {}
    validation_report_text = "\n".join(
        paragraph
        for document in ctx.documents if document["port"] == "validation_report"
        for paragraph in document["paragraphs"]
    )

    def validate(proposal: dict) -> None:
        plan = resolve_measurement_plan(proposal, layout, frame, sheet)
        if plan.reported_raw is not None:
            verify_reported_citation(plan.reported_raw, validation_report_text)
        scored, km = evaluate(frame, layout, plan)
        # Проекция — часть валидации: конфликт имён колонок возвращается
        # модели как repair, а не роняет прогон.
        published = publish_umr(scored, layout, plan)
        resolved.update(plan=plan, km=km, published=published)

    base_messages = metric_messages(
        _column_inventory(sheet, frame, layout), ctx.documents,
        {"grouping": layout.grouping["kind"],
         "first_data_row": layout.first_data_row,
         "last_data_row": layout.last_data_row},
    )
    request_structured(client, base_messages, METRIC_SCHEMA, "metric",
                       validate_extra=validate)
    if resolved["km"]["percent_domain_columns"]:
        journal.warning(
            "score_domain_percent",
            "оценки в колонках "
            f"{resolved['km']['percent_domain_columns']} заданы в процентных "
            "пунктах при шкале percent — приведены к долям",
        )
    return resolved["plan"], resolved["km"], resolved["published"]
