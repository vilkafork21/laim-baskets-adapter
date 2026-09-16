"""Оркестрация независимых задач L (структура), S (оценки) и K (baseline).

Разметка обязана собрать обязательные поля спеки (иначе SpecError — нода
падает); план метрики деградирует: его провал оставляет корзину публикуемой.
КМ публикуется только заявленной в отчёте о валидации: цитата значения
проверяется кодом, отсутствие значения в отчёте — дефект артефактов.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from .. import defaults
from ..context import build_run_context as build_run_context
from ..context import check_report_identity as check_report_identity
from ..errors import (
    BasketError,
    LayoutError,
    LlmError,
    SpecError,
    StructuredOutputError,
)
from ..evidence.workbook import workbook_evidence
from ..journal import Journal
from ..metric.baseline import BaselineResult, failed_baseline, select_baseline
from ..metric.engine import evaluate
from ..metric.resolve import (
    resolve_measurement_plan,
    vertically_merged_source,
)
from ..models import MeasurementPlan, ResolvedLayout, RunContext
from ..publish import PublishedUmr, publish_umr
from ..resolve import resolve_layout
from ..transform.canon import build_canon
from ..transform.grouping import apply_grouping
from ..transform.values import blank
from .client import request_structured
from .prompts import baseline_batches, layout_messages, metric_messages
from .schemas import BASELINE_SCHEMA, LAYOUT_SCHEMA, METRIC_SCHEMA

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayoutOutcome:
    proposal: dict
    layout: ResolvedLayout
    frame: pd.DataFrame
    conversion: dict


def _materialize(
    proposal: dict, ctx: RunContext, pinned_sheet: str, rejected_sheets: frozenset[str]
):
    layout = resolve_layout(proposal, ctx.sheets, ctx.basket_id, pinned_sheet, rejected_sheets)
    sheet = ctx.sheets[layout.sheet_name]
    grouped = apply_grouping(sheet, layout.region, layout.transform_config())
    frame, conversion = build_canon(grouped, layout.region, layout.transform_config())
    return layout, frame, conversion


def run_layout(
    client, ctx: RunContext, journal: Journal, pinned_sheet: str, rejected_sheets: frozenset[str]
) -> LayoutOutcome:
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
                error=error,
                proposal=copy.deepcopy(proposal),
                layout=layout,
                frame=frame,
                conversion=conversion,
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
            blank_query = set(missing_values.get("input_query", {}).get("row_positions", []))
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
                and all(
                    violation["reason"] == "blank_group"
                    for violation in validation["context_violations"]
                )
            )
            bad = blank_query | blank_group
            if row_scoped and bad and len(bad) < len(frame):
                keep = [position not in bad for position in range(len(frame))]
                kept = frame.loc[keep].reset_index(drop=True)
                dropped = {
                    kind: sorted(
                        int(frame["source_row_id"].iloc[position]) for position in positions
                    )
                    for kind, positions in (
                        ("blank_input_query", blank_query),
                        ("blank_group", blank_group - blank_query),
                    )
                    if positions
                }
                fixed = copy.deepcopy(conversion)
                fixed["row_accounting"]["canon_rows"] = len(kept)
                fixed["row_accounting"]["dropped_invalid_rows"] = sorted(
                    row for rows in dropped.values() for row in rows
                )
                fixed["umr_validation"]["status"] = "passed"
                fixed["umr_validation"]["missing_required_values"] = {}
                fixed["umr_validation"]["context_violations"] = []
                recoverable.update(
                    error=error,
                    proposal=copy.deepcopy(proposal),
                    layout=layout,
                    frame=kept,
                    conversion=fixed,
                    dropped=dropped,
                )
            raise error
        resolved.update(proposal=proposal, layout=layout, frame=frame, conversion=conversion)

    try:
        request_structured(
            client,
            layout_messages(evidence, ctx.documents, pinned_sheet, rejected_sheets),
            LAYOUT_SCHEMA,
            "layout",
            validate_extra=validate,
        )
    except (LayoutError, StructuredOutputError, LlmError) as exc:
        if not recoverable:
            details = dict(exc.details)
            if attempted.get("sheet"):
                # Лист последней попытки нужен pipeline для запасного листа.
                details["sheet"] = attempted["sheet"]
            raise SpecError(
                "Этап L. Ожидалось: канон с непустым input_query. "
                f"Получено: обязательные поля спеки не собраны после repair: {exc}. "
                "Действие: проверьте колонку запроса, границы данных и sheet_name.",
                **details,
            ) from exc
        # Repair не помог, но кандидат публикуем: отброс меньшинства строк —
        # деградация с учётом, а не падение ноды.
        dropped_rows = sum(len(rows) for rows in recoverable["dropped"].values())
        kept_rows = recoverable["frame"]["source_row_id"].nunique()
        if dropped_rows > (dropped_rows + kept_rows) * defaults.MAX_DROPPED_ROW_SHARE:
            recovered_error = recoverable["error"]
            raise SpecError(
                "Разметка отбрасывает больше половины строк корзины — обязательные "
                f"поля спеки не собраны: {recovered_error}",
                dropped_rows=dropped_rows,
                kept_rows=int(kept_rows),
                **recovered_error.details,
            ) from recovered_error
        for kind, rows in recoverable["dropped"].items():
            journal.dropped(kind, rows)
        resolved.update(
            proposal=recoverable["proposal"],
            layout=recoverable["layout"],
            frame=recoverable["frame"],
            conversion=recoverable["conversion"],
        )
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
        counts = Counter(str(value) for value in present)
        categorical = len(counts) <= 64 and all(len(value) <= 200 for value in counts)
        formula_rows = list(layout.formula_rows.get(column_id, ()))
        column0 = layout.region.column_letters.index(column_id)
        result.append(
            {
                "blank_count": len(values) - len(present),
                "formula_count": len(formula_rows),
                "formula_samples": [
                    {"row": row, "formula": sheet.formulas.get((row - 1, column0), "")}
                    for row in formula_rows[:3]
                ],
                "formula_rows_truncated": len(formula_rows) > 20,
                "unique_values": dict(counts) if categorical else {},
                "unique_values_complete": categorical,
                "unique_count": len(counts),
                "column_id": column_id,
                "canonical_raw_name": name,
                "non_null": len(present),
                "samples": [str(value)[:120] for value in present[:3]],
                "formula_rows": formula_rows[:20],
                "vertically_merged": vertically_merged_source(sheet, layout, column_id),
            }
        )
    return result


def run_metric(
    client, ctx: RunContext, outcome: LayoutOutcome, journal: Journal
) -> tuple[MeasurementPlan, dict, PublishedUmr]:
    layout, frame = outcome.layout, outcome.frame
    sheet = ctx.sheets[layout.sheet_name]
    resolved: dict = {}

    def validate(proposal: dict) -> None:
        plan = resolve_measurement_plan(proposal, layout, frame, sheet)
        scored, km = evaluate(frame, layout, plan)
        # Проекция — часть валидации: конфликт имён колонок возвращается
        # модели как repair, а не роняет прогон.
        published = publish_umr(scored, layout, plan)
        resolved.update(plan=plan, km=km, published=published)

    base_messages = metric_messages(
        _column_inventory(sheet, frame, layout),
        ctx.documents,
        {
            "grouping": layout.grouping["kind"],
            "first_data_row": layout.first_data_row,
            "last_data_row": layout.last_data_row,
        },
    )
    request_structured(client, base_messages, METRIC_SCHEMA, "metric", validate_extra=validate)
    if resolved["plan"].missing_values:
        journal.warning(
            "declared_missing_values",
            f"Явные нечисловые маркеры пропуска: {resolved['plan'].missing_values}; "
            "проверьте основание исключений по методике.",
        )
    if resolved["km"]["percent_domain_columns"]:
        journal.warning(
            "score_domain_percent",
            "оценки в колонках "
            f"{resolved['km']['percent_domain_columns']} заданы в процентных "
            f"пунктах при шкале {resolved['plan'].scale} — приведены к долям",
        )
    formula_sources = {
        source["column_id"]: list(layout.formula_rows[source["column_id"]])
        for source in resolved["plan"].sources
        if source["column_id"] in layout.formula_rows
    }
    if formula_sources:
        journal.warning(
            "formula_score_cached",
            f"Этап S: использован кеш построчных формул "
            f"{ {column: len(rows) for column, rows in formula_sources.items()} } (колонка: число строк); "
            "формулы не пересчитывались. Перед загрузкой пересчитайте и сохраните Excel.",
        )
    return resolved["plan"], resolved["km"], resolved["published"]


def run_baseline(
    client, ctx: RunContext, selected_sheet: str, plan: MeasurementPlan | None = None
) -> BaselineResult:
    report = next(document for document in ctx.documents if document["port"] == "validation_report")
    candidates, failures = [], []
    for index, messages in enumerate(
        baseline_batches(report, list(ctx.sheets), plan.metric_name if plan else ""), 1
    ):
        try:
            candidates.extend(request_structured(client, messages, BASELINE_SCHEMA, "baseline"))
        except BasketError as exc:
            failures.append(f"часть {index}: {exc.reason_code}: {exc}")
    # Перекрытие длинного абзаца не размножает одно и то же упоминание.
    candidates = list(
        {json.dumps(c, sort_keys=True, ensure_ascii=False): c for c in candidates}.values()
    )
    result = select_baseline(
        candidates,
        report["paragraphs"],
        selected_sheet=selected_sheet,
        metric_name=plan.metric_name if plan else "",
        scale=plan.scale if plan else None,
    )
    if failures:
        return failed_baseline(
            "Отчёт извлечён не полностью: " + "; ".join(failures),
            candidates=result.candidates,
            rejected=result.rejected,
        )
    return result
