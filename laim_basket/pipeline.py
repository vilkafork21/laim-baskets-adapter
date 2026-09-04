"""Единственный продовый путь: пакет -> разметка -> канон -> КМ -> публикация.

Этапы пишутся в журнал (logging + порт km_result). Провал плана метрики —
деградация not_computable с публикацией корзины; падение ноды оставлено
битому пакету, недоступной LLM и несобранным обязательным полям спеки.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

from .config import llm_config
from .errors import (
    AmbiguousBaselineError,
    BasketError,
    MeasurementPlanError,
    NotEvaluableError,
    SpecError,
    StructuredOutputError,
)
from .export.umr import export_umr_workbook
from .journal import Journal
from .llm import tasks
from .llm.client import LlmClient
from .models import RunResult
from .publish import publish_umr

logger = logging.getLogger(__name__)


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    _json(temporary, payload)
    os.replace(temporary, path)


def reset_out(out_dir: str | Path) -> None:
    out = Path(out_dir)
    shutil.rmtree(out / "debug", ignore_errors=True)
    for pattern in ("umr_*.xlsx", "run_report.json", "km.json", "failure.json",
                     ".*.tmp", ".*.tmp.xlsx"):
        for stale in out.glob(pattern):
            stale.unlink(missing_ok=True)


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _km_summary(km: dict | None) -> dict | None:
    """Блок km отчёта laim-run-report.v1 — сводка для человека и триажа."""
    if not km or not isinstance(km.get("main_metric"), dict):
        return None
    metric = km["main_metric"]
    return {
        "name": metric["name"],
        "value": metric["value"],
        "recomputed_value": metric["recomputed_value"],
        "scale": metric["scale"],
        "reconciliation": km["reconciliation"]["status"],
        "coverage": km["coverage"],
    }


def _not_computable(basket_id: str, reason: BasketError) -> dict[str, object]:
    """Внутренняя сводка метрики при недостроенном плане (не контракт)."""
    return {
        "status": "not_computable",
        "basket_id": basket_id,
        "reason": str(reason),
        "reason_code": reason.reason_code,
        "details": reason.details,
        "main_metric": None,
        "reconciliation": {"status": "not_evaluable", "difference": None},
        "threshold_verdict": None,
    }


def run_package(
    input_path: str | Path,
    out_dir: str | Path,
    client=None,
    sheet_name: str = "",
    run_context: dict[str, object] | None = None,
) -> RunResult:
    """sheet_name задаёт обязательный лист корзины (книги с листами нескольких
    агентов различимы только оператором); пустое значение — автоопределение."""
    root = Path(out_dir)
    reset_out(root)
    debug = root / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    journal = Journal()
    started = time.monotonic()
    context = tasks.build_run_context(input_path, run_context)
    journal.set_inputs(context.file_hashes)
    tasks.check_report_identity(context, journal)
    journal.stage("read", "ok", _ms(started))
    llm = client or LlmClient(llm_config(), debug)

    plan, km, published = None, None, None
    status = "computed"
    rejected: frozenset[str] = frozenset()
    # Канон первого листа, на котором не собрался план: если запасной лист
    # не размечается вовсе, корзина публикуется с него как not_computable.
    fallback: tuple[tasks.LayoutOutcome, BasketError] | None = None
    for attempt in (1, 2):
        stage_started = time.monotonic()
        try:
            outcome = tasks.run_layout(llm, context, journal, sheet_name, rejected)
        except SpecError as exc:
            failed_sheet = exc.details.get("sheet")
            if fallback is not None:
                journal.stage("layout", "degraded", _ms(stage_started))
                journal.warning(exc.reason_code,
                                f"запасной лист не размечен: {exc}")
                outcome, cause = fallback
                journal.decision(sheet=outcome.layout.sheet_name,
                                 grouping=outcome.layout.grouping["kind"])
                plan, km = None, _not_computable(context.basket_id, cause)
                published = publish_umr(outcome.frame, outcome.layout, None)
                status = "not_computable"
                break
            if sheet_name or attempt == 2 or len(context.sheets) < 2 or not failed_sheet:
                raise
            # Симметрия с этапом метрики: несобираемый лист отвергается,
            # разметка пробуется на запасном.
            journal.stage("layout", "degraded", _ms(stage_started))
            journal.warning(
                exc.reason_code,
                f"разметка не собрана на листе {failed_sheet!r}: {exc}",
            )
            rejected = frozenset({failed_sheet})
            continue
        journal.stage("layout", "ok", _ms(stage_started))
        journal.decision(sheet=outcome.layout.sheet_name,
                         grouping=outcome.layout.grouping["kind"])
        logger.info(
            "Лист %r: шапка %s, данные строки %d-%d, роли %s",
            outcome.layout.sheet_name, list(outcome.layout.header_rows),
            outcome.layout.first_data_row, outcome.layout.last_data_row,
            {role: value["source"] for role, value in outcome.layout.roles.items()
             if isinstance(value, dict) and "source" in value},
        )
        _json(debug / "layout_proposal.json", outcome.proposal)
        _json(debug / "resolved_layout.json", outcome.layout.to_dict())
        _json(debug / "conversion.json", outcome.conversion)

        stage_started = time.monotonic()
        try:
            plan, km, published = tasks.run_metric(llm, context, outcome, journal)
            journal.stage("metric", "ok", _ms(stage_started))
            journal.decision(assessment_mode=plan.assessment_mode,
                             metric=plan.metric_name, method=plan.method,
                             reducer=plan.reducer)
            _json(debug / "measurement_plan.json", plan.to_dict())
            break
        except (MeasurementPlanError, NotEvaluableError, StructuredOutputError) as exc:
            journal.stage("metric", "degraded", _ms(stage_started))
            journal.warning(
                exc.reason_code,
                f"план не построен на листе {outcome.layout.sheet_name!r}: {exc}",
            )
            if (
                not isinstance(exc, AmbiguousBaselineError)
                and not sheet_name
                and attempt == 1
                and len(context.sheets) > 1
            ):
                rejected = frozenset({outcome.layout.sheet_name})
                fallback = (outcome, exc)
                continue
            # Корзина публикуется всегда, когда канон собран.
            plan, km = None, _not_computable(context.basket_id, exc)
            published = publish_umr(outcome.frame, outcome.layout, None)
            status = "not_computable"
            break
    if plan is not None and plan.reported_value is None:
        # Дефект артефактов: km_result обязан согласоваться с monitoring_metric,
        # который уйдёт как not_computable / official_baseline_missing.
        status = "not_computable"
        journal.warning(
            "official_baseline_missing",
            "отчёт о валидации не объявляет КМ — корзина публикуется без значения",
        )
    _json(debug / "publication.json", published.to_dict())

    excel_name = f"umr_{context.basket_id}.xlsx"
    temporary_excel = root / f".{excel_name}.{uuid.uuid4().hex}.tmp.xlsx"
    root.mkdir(parents=True, exist_ok=True)
    try:
        export_umr_workbook(published, temporary_excel)
        os.replace(temporary_excel, root / excel_name)
    finally:
        temporary_excel.unlink(missing_ok=True)

    journal.set_llm(model=llm.config.model, structured_output=llm.structured_output,
                    calls=llm.calls, repair_turns=llm.repair_turns,
                    transport_retries=llm.transport_retries)
    report = journal.report(basket_id=context.basket_id, status=status,
                            km=_km_summary(km))
    _atomic_json(root / "run_report.json", report)
    logger.info("Итог прогона: статус %s, строк UMR %d, файл %s",
                status, len(published.frame), excel_name)
    return RunResult(
        status=status,
        umr=published,
        km=km,
        excel_name=excel_name,
        measurement_plan=plan,
        report=report,
    )
