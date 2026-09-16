"""Контракт monitoring_metric v2; не читает файлы и не вызывает модель."""

from decimal import Decimal

from .errors import PackageError
from .metric.nonadditive import METHODS as NONADDITIVE_METHODS
from .models import RunResult


def _normalized(value: object, scale: str) -> tuple[float, str]:
    number = Decimal(str(value))
    if scale == "percent":
        return float(number / Decimal(100)), "ratio"
    if scale in {"ratio", "raw"}:
        return float(number), scale
    raise PackageError("MeasurementPlan содержит неизвестную шкалу", scale=scale)


def monitoring_metric(result: RunResult) -> dict[str, object]:
    plan = result.measurement_plan
    contract = {
        "contract_version": "laim-monitoring-metric.v2",
        "umr_version": "laim-umr.v2",
    }
    if plan is None:
        return {
            **contract,
            "status": "not_computable",
            "basket_id": result.km.get("basket_id"),
            "reason": result.km.get("reason", "MeasurementPlan не построен"),
            "reason_code": result.km.get("reason_code"),
        }
    # Отсутствие КМ в отчёте о валидации — специфичная деградация (со своим
    # baseline-блоком), проверяется до общего not_computable по статусу.
    if plan.reported_value is None and isinstance(result.km.get("main_metric"), dict):
        recomputed_value, recomputed_scale = _normalized(
            result.km["main_metric"]["recomputed_value"], plan.scale
        )
        return {
            **contract,
            "status": "not_computable",
            "basket_id": plan.basket_id,
            "assessment_mode": plan.assessment_mode,
            "reason": result.km.get(
                "reason", "Этап K: Validation report не содержит официальный baseline"
            ),
            "reason_code": result.km.get("reason_code", "official_baseline_missing"),
            "baseline": {
                "value": None,
                "scale": recomputed_scale,
                "value_source": None,
                "reported_value": None,
                "reported_scale": None,
                "recomputed_value": recomputed_value,
                "reconciliation": result.km["reconciliation"]["status"],
            },
        }
    if result.status != "computed" or not isinstance(result.km.get("main_metric"), dict):
        return {
            **contract,
            "status": "not_computable",
            "basket_id": plan.basket_id,
            "assessment_mode": plan.assessment_mode,
            "reason": result.km.get("reason", "КМ не вычислена"),
            "reason_code": result.km.get("reason_code"),
        }
    nonadditive = plan.method in NONADDITIVE_METHODS
    metric = result.km["main_metric"]
    recomputed_value, recomputed_scale = _normalized(metric["recomputed_value"], plan.scale)
    baseline_value, baseline_scale = _normalized(plan.reported_value, plan.scale)
    return {
        **contract,
        "status": "computed",
        "basket_id": plan.basket_id,
        "name": plan.metric_name,
        "score_column": None if nonadditive else "main_metric",
        **(
            {"score_scope": "dataset", "required_capabilities": ["laim.nonadditive-metrics.v1"]}
            if nonadditive
            else {}
        ),
        "assessment_mode": plan.assessment_mode,
        "scoring": {
            "method": plan.method,
            **({"metric_options": dict(plan.metric_options)} if nonadditive else {}),
            "sources": [
                {
                    "source_id": f"source_{index}",
                    "column_name": result.umr.published_columns[source["column_id"]],
                    "role": source["role"],
                    # Метрики публикуются уже нормализованными числами (value_map
                    # и инверсия применены), поэтому контракт всегда direct/numeric;
                    # labels сравниваются как labels.
                    "normalization": "label"
                    if source["role"] in ("prediction", "target")
                    else "numeric",
                    "polarity": "direct",
                }
                for index, source in enumerate(plan.sources, start=1)
            ],
            "missing_policy": plan.missing_policy,
            "majority_denominator": plan.majority_denominator,
        },
        "aggregation": {
            "method": plan.method if nonadditive else plan.reducer,
            **(
                {
                    "sample_weighting": "frequency"
                    if plan.reducer == "frequency_weighted_mean"
                    else "uniform"
                }
                if nonadditive
                else {}
            ),
            "weight_column": "input_query_count"
            if plan.reducer == "frequency_weighted_mean"
            else None,
        },
        "baseline": {
            "value": baseline_value,
            "scale": baseline_scale,
            "value_source": "validation_report",
            "reported_value": float(plan.reported_value),
            "reported_scale": plan.scale,
            "recomputed_value": recomputed_value,
            "reconciliation": result.km["reconciliation"]["status"],
        },
        "primary_validation": {
            "threshold": float(plan.threshold) if plan.threshold is not None else None,
            "comparator": plan.comparator,
            "scale": plan.scale,
            "verdict": result.km.get("threshold_verdict"),
            "affects_monitoring": False,
        },
        "evidence": {key: list(value) for key, value in plan.evidence.items()},
    }
