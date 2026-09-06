"""Расчёт КМ по плану на канонической корзине.

Единицы оценки и формулу считает общий пакет laim_monitoring — тот же код,
которым km-dynamic считает КМ на мониторинге. Здесь остаётся только то, что
касается отчёта о валидации: шкала, сверка с заявленным значением, порог.
"""

from __future__ import annotations

from decimal import Decimal

from laim_monitoring import (
    FormulaError,
    MonitoringContractError,
    broadcast_scores,
    evaluate_formula,
    unit_scores,
    unitize,
)

from ..errors import NotEvaluableError
from ..measurement import reported_quantum
from ..models import MeasurementPlan, ResolvedLayout


def plan_contract(plan: MeasurementPlan, layout: ResolvedLayout) -> dict[str, object]:
    """Контракт для расчёта на корзине: входы указывают на сырые колонки листа."""
    return {
        "assessment_mode": plan.assessment_mode,
        "formula": plan.formula,
        "inputs": [
            {"name": item["name"], "column": layout.column_names[item["column_id"]], "judged": item["judged"]}
            for item in plan.inputs
        ],
    }


def _in_scale(value: Decimal, scale: str) -> Decimal:
    return value * 100 if scale == "percent" else value


def evaluate(frame, layout: ResolvedLayout, plan: MeasurementPlan) -> tuple[object, dict[str, object]]:
    contract = plan_contract(plan, layout)
    try:
        units = unitize(frame, contract)
        result = evaluate_formula(units, contract)
        scores = unit_scores(units, contract)
    except (MonitoringContractError, FormulaError) as exc:
        raise NotEvaluableError(f"Расчёт КМ: {exc}") from exc

    recomputed = _in_scale(Decimal(str(result["value"])), plan.scale)
    reported = plan.reported_value
    final_value = reported if reported is not None else recomputed
    value_source = "validation_report" if reported is not None else "recomputed"

    verdict = None
    if plan.threshold is not None:
        passed = final_value >= plan.threshold if plan.comparator == ">=" else final_value <= plan.threshold
        verdict = "passed" if passed else "failed"

    reconciliation, difference = "not_applicable", None
    if reported is not None:
        difference = reported - recomputed
        reconciliation = "match" if abs(difference) <= reported_quantum(plan) else "mismatch"

    scored_frame = broadcast_scores(frame, units, scores).drop(columns=["assessment_unit_id"])
    km = {
        "report_version": "laim-km.v3",
        "status": "computed",
        "value_source": value_source,
        "recomputed_value": float(recomputed),
        "coverage": {key: result[key] for key in ("total_units", "scored_units", "excluded_units", "weight_sum")},
        "reconciliation": {
            "status": reconciliation,
            "difference": float(difference) if difference is not None else None,
        },
        "evidence": {key: list(value) for key, value in plan.evidence.items()},
        "threshold_verdict": verdict,
        "main_metric": {
            "name": plan.metric_name,
            "value": float(final_value),
            "recomputed_value": float(recomputed),
            "value_source": value_source,
            "formula": plan.formula,
            "inputs": contract["inputs"],
            "assessment_mode": plan.assessment_mode,
            "scale": plan.scale,
            "precision": plan.precision,
            "threshold": float(plan.threshold) if plan.threshold is not None else None,
            "comparator": plan.comparator,
            "threshold_verdict": verdict,
            "reconciliation_status": reconciliation,
            "n_units": result["scored_units"],
            "units_dropped_nan_score": result["excluded_units"],
        },
    }
    return scored_frame, km
