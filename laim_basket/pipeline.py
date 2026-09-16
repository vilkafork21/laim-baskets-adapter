"""Независимые этапы L -> S и K; провалы S/K не меняют собранную корзину."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

from .config import llm_config
from .errors import BasketError, ScorePlanError, SpecError
from .export.umr import export_umr_workbook
from .journal import Journal
from .llm import tasks
from .llm.client import LlmClient
from .metric.baseline import attach_baseline, failed_baseline
from .metric.engine import reconcile
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
    for pattern in (
        "umr_*.xlsx",
        "run_report.json",
        "km.json",
        "failure.json",
        ".*.tmp",
        ".*.tmp.xlsx",
    ):
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
    agent_ci: str = "",
) -> RunResult:
    """sheet_name задаёт обязательный лист корзины (книги с листами нескольких
    агентов различимы только оператором); пустое значение — автоопределение."""
    root = Path(out_dir)
    reset_out(root)
    debug = root / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    journal = Journal()
    started = time.monotonic()
    context = tasks.build_run_context(input_path, agent_ci=agent_ci)
    journal.set_inputs(context.file_hashes)
    tasks.check_report_identity(context, journal)
    journal.stage("read", "ok", _ms(started))
    llm = client or LlmClient(llm_config(), debug)

    # L: запасной лист допустим только до получения канона.
    rejected: frozenset[str] = frozenset()
    for attempt in (1, 2):
        stage_started = time.monotonic()
        try:
            outcome = tasks.run_layout(llm, context, journal, sheet_name, rejected)
        except SpecError as exc:
            journal.stage("layout", "degraded", _ms(stage_started))
            failed_sheet = exc.details.get("sheet")
            if sheet_name or attempt == 2 or len(context.sheets) < 2 or not failed_sheet:
                raise
            journal.warning(exc.reason_code, str(exc))
            rejected = frozenset({failed_sheet})
            continue
        journal.stage("layout", "ok", _ms(stage_started))
        break
    journal.decision(
        sheet=outcome.layout.sheet_name,
        grouping=outcome.layout.grouping["kind"],
        basket_id_source="agent_ci" if (agent_ci or "").strip() else "package_name",
    )
    _json(debug / "layout_proposal.json", outcome.proposal)
    _json(debug / "resolved_layout.json", outcome.layout.to_dict())
    _json(debug / "conversion.json", outcome.conversion)

    # S: сохраняем оценки даже при последующем отказе K.
    plan, km, published, score_error = None, None, None, None
    stage_started = time.monotonic()
    try:
        plan, km, published = tasks.run_metric(llm, context, outcome, journal)
        journal.stage("metric", "ok", _ms(stage_started))
        journal.decision(
            assessment_mode=plan.assessment_mode,
            metric=plan.metric_name,
            method=plan.method,
            reducer=plan.reducer,
        )
    except BasketError as exc:
        reason = (
            "Этап S. Ожидалось: исполнимый план оценки единиц на выбранном листе. "
            f"Получено: {exc.reason_code}: {exc}. Действие: проверьте score-колонки, "
            "политику пропусков, кеш формул и методику в документах; повторите запуск. "
            "Лист не изменён; корзина опубликована без main_metric."
        )
        score_error = ScorePlanError(reason, cause=exc.to_dict(), stage="S")
        km = _not_computable(context.basket_id, score_error)
        published = publish_umr(outcome.frame, outcome.layout, None)
        journal.stage("metric", "degraded", _ms(stage_started))
        journal.warning(score_error.reason_code, reason)

    # K выполняется и после отказа S; ни recomputed_value, ни данные scores в LLM не уходят.
    stage_started = time.monotonic()
    baseline = None
    try:
        baseline = tasks.run_baseline(llm, context, outcome.layout.sheet_name, plan)
        if plan is not None and baseline.state == "declared":
            plan = attach_baseline(plan, baseline)
            km = reconcile(km, plan)
    except BasketError as exc:
        previous = baseline
        baseline = failed_baseline(
            f"{exc.reason_code}: {exc}",
            ambiguous=previous is not None,
            candidates=previous.candidates if previous is not None else (),
            rejected=previous.rejected if previous is not None else (),
        )
    journal.stage(
        "baseline", "ok" if baseline.state == "declared" else "degraded", _ms(stage_started)
    )
    for warning in baseline.warnings:
        journal.warning(warning["code"], warning["message"])
    if baseline.reason_code:
        journal.warning(baseline.reason_code, baseline.reason)
    status = "computed" if plan is not None and baseline.state == "declared" else "not_computable"
    if score_error is None and status == "not_computable":
        km.update(
            status=status,
            basket_id=context.basket_id,
            reason_code=baseline.reason_code,
            reason=baseline.reason,
        )
    if plan is not None:
        _json(debug / "measurement_plan.json", plan.to_dict())
    _json(debug / "baseline.json", baseline.to_dict())
    _json(debug / "publication.json", published.to_dict())

    excel_name = f"umr_{context.basket_id}.xlsx"
    temporary_excel = root / f".{excel_name}.{uuid.uuid4().hex}.tmp.xlsx"
    root.mkdir(parents=True, exist_ok=True)
    try:
        export_umr_workbook(published, temporary_excel)
        os.replace(temporary_excel, root / excel_name)
    finally:
        temporary_excel.unlink(missing_ok=True)

    coverage = km.get("coverage", {})
    if coverage.get("excluded_units", 0):
        journal.warning(
            "score_coverage_partial",
            f"Оценены {coverage['scored_units']} из {coverage['total_units']} единиц; "
            "проверьте исключения и знаменатель по методике.",
        )
    if km["reconciliation"]["status"] == "mismatch":
        journal.warning(
            "reconciliation_mismatch",
            f"Официальная КМ {km['main_metric']['value']}, пересчёт "
            f"{km['recomputed_value']}; проверьте версию корзины, методику и кеш. "
            "Статус computed не является production-допуском.",
        )

    journal.set_llm(
        model=llm.config.model,
        structured_output=llm.structured_output,
        calls=llm.calls,
        repair_turns=llm.repair_turns,
        transport_retries=llm.transport_retries,
    )
    report = journal.report(basket_id=context.basket_id, status=status, km=_km_summary(km))
    report["baseline"] = baseline.to_dict()
    if status != "computed":
        report["reason_code"], report["reason"] = km["reason_code"], km["reason"]
    _atomic_json(root / "run_report.json", report)
    logger.info(
        "Итог прогона: статус %s, строк UMR %d, файл %s", status, len(published.frame), excel_name
    )
    return RunResult(
        status=status,
        umr=published,
        km=km,
        excel_name=excel_name,
        measurement_plan=plan,
        report=report,
    )
