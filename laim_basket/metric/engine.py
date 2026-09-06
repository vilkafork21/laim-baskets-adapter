"""Расчёт КМ по MeasurementPlan.

Ячейки корзины нормализуются (value_map, инверсия, проценты), строки
собираются в единицы оценки (реплика или диалог), а само значение считает
общий вычислитель laim_monitoring — та же формула, которой km-dynamic считает
КМ на мониторинге. Поэтому baseline и мониторинг сопоставимы по построению.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from laim_monitoring import (
    FormulaError,
    MonitoringContractError,
    contract_formula,
    formula_columns,
    parse_formula,
    unit_scores,
)

from ..errors import NotEvaluableError
from ..measurement import decimal_value, reported_quantum
from ..models import MeasurementPlan, ResolvedLayout
from ..transform.values import blank as _blank, normalize_key


def _numeric(value: object) -> Decimal | None:
    if _blank(value):
        return None
    if isinstance(value, bool):
        raise NotEvaluableError("Boolean нельзя использовать как неявный numeric score")
    text = str(value).strip()
    percent = text.endswith("%")
    result = decimal_value(text.rstrip("%").strip())
    return result / 100 if percent else result


def _normalizer(source: dict[str, object]):
    normalization = source["normalization"]
    if normalization == "label":
        return lambda value: None if _blank(value) else normalize_key(value)
    if normalization == "numeric":
        base = _numeric
    else:
        lookup = {normalize_key(key): decimal_value(value) for key, value in normalization.items()}

        def base(value: object) -> Decimal | None:
            if _blank(value):
                return None
            key = normalize_key(value)
            if key not in lookup:
                raise NotEvaluableError("value_map не покрывает фактическое значение", value=str(value)[:80])
            return lookup[key]

    if source["polarity"] == "direct":
        return base

    def inverted(value: object) -> Decimal | None:
        normalized = base(value)
        if normalized is None:
            return None
        if normalized not in (Decimal(0), Decimal(1)):
            raise NotEvaluableError("polarity=inverted допустима только для бинарного score")
        return Decimal(1) - normalized

    return inverted


def source_values(frame, layout: ResolvedLayout, plan: MeasurementPlan) -> dict[str, list[object]]:
    result = {}
    for source in plan.sources:
        column_id = source["column_id"]
        normalizer = _normalizer(source)
        result[column_id] = [normalizer(value) for value in frame[layout.column_names[column_id]].tolist()]
    return result


def _unit_records(frame, values: dict[str, list[object]], plan: MeasurementPlan) -> list[dict[str, object]]:
    if plan.evaluation_unit == "turn":
        return [
            {
                "values": {column: column_values[index] for column, column_values in values.items()},
                "weight": decimal_value(frame["input_query_count"].iloc[index]),
                "rows": [index],
            }
            for index in range(len(frame))
        ]
    missing = [name for name in ("reference_group_id", "turn_index") if name not in frame]
    if missing:
        raise NotEvaluableError(
            "Dialogue КМ требует явные group и turn order",
            missing_columns=missing,
        )
    groups: dict[object, list[int]] = {}
    group_values = frame["reference_group_id"].tolist()
    for index, group in enumerate(group_values):
        groups.setdefault(group, []).append(index)
    records = []
    for group, indexes in groups.items():
        unit_values = {}
        for column, column_values in values.items():
            present = {column_values[index] for index in indexes if column_values[index] is not None}
            if len(present) > 1:
                raise NotEvaluableError(
                    "Источник КМ не константен внутри dialogue",
                    group=str(group), column_id=column,
                )
            unit_values[column] = next(iter(present)) if present else None
        if plan.reducer == "frequency_weighted_mean":
            weights = {decimal_value(frame["input_query_count"].iloc[index]) for index in indexes}
            if len(weights) != 1:
                raise NotEvaluableError("Weight не константен внутри dialogue", group=str(group))
            weight = next(iter(weights))
        else:
            weight = Decimal(1)
        records.append({"values": unit_values, "weight": weight, "rows": indexes})
    return records


def plan_contract(plan: MeasurementPlan) -> dict[str, object]:
    """Часть контракта monitoring_metric, нужная формуле: источники и её текст.

    Внутри адаптера source_id источника — это column_id корзины; имя входа в
    формуле (`name`) одинаково здесь и в опубликованном контракте.
    """
    contract = {
        "scoring": {
            "method": plan.method,
            "sources": [
                {
                    "source_id": source["column_id"],
                    "column_name": source["column_id"],
                    "name": source["name"],
                    "role": source["role"],
                    "normalization": "label" if source["role"] in ("prediction", "target") else "numeric",
                    "polarity": "direct",
                }
                for source in plan.sources
            ],
            "missing_policy": plan.missing_policy,
            "majority_denominator": plan.majority_denominator,
        },
        "aggregation": {"method": plan.reducer},
    }
    contract["formula"] = plan.formula or contract_formula(contract)
    return contract


def _units_frame(records: list[dict[str, object]], values: dict[str, list[object]]) -> pd.DataFrame:
    units = pd.DataFrame({
        column_id: [record["values"][column_id] for record in records] for column_id in values
    })
    units["input_query_count"] = [float(record["weight"]) for record in records]
    return units


def _published_scale(value: Decimal, scale: str) -> Decimal:
    return value * 100 if scale == "percent" else value


def evaluate(frame, layout: ResolvedLayout, plan: MeasurementPlan) -> tuple[object, dict[str, object]]:
    values = source_values(frame, layout, plan)
    records = _unit_records(frame, values, plan)
    units = _units_frame(records, values)
    contract = plan_contract(plan)
    try:
        formula = parse_formula(contract["formula"])
        columns = formula_columns(units, contract)
        if plan.missing_policy == "fail":
            blank = [name for name in formula.inputs if columns[name].isna().any()]
            if blank:
                raise NotEvaluableError("В источниках КМ есть пропуски, а missing_policy=fail", inputs=blank)
        scores = unit_scores(units, contract)
        value = formula.evaluate(columns)
    except (FormulaError, MonitoringContractError) as exc:
        raise NotEvaluableError(f"Формула КМ: {exc}") from exc
    if pd.isna(value):
        raise NotEvaluableError("Ни одной оцененной единицы")
    scored_mask = (
        scores.notna() if formula.unit_expression() is not None
        else pd.concat([columns[name] for name in formula.inputs], axis=1).notna().all(axis=1)
    )
    if not scored_mask.any():
        raise NotEvaluableError("Ни одной оцененной единицы")
    weighted = plan.reducer == "frequency_weighted_mean"
    weights = units["input_query_count"] if weighted else pd.Series(1.0, index=units.index)
    if weighted and (weights[scored_mask] <= 0).any():
        raise NotEvaluableError("Вес должен быть положительным")
    total_weight = Decimal(str(float(weights[scored_mask].sum())))
    recomputed = Decimal(str(value))
    published_recomputed = _published_scale(recomputed, plan.scale)
    final_value = plan.reported_value if plan.reported_value is not None else published_recomputed
    source = "validation_report" if plan.reported_value is not None else "recomputed"
    verdict = None
    if plan.threshold is not None:
        if plan.comparator == ">=":
            verdict = "passed" if final_value >= plan.threshold else "failed"
        else:
            verdict = "passed" if final_value <= plan.threshold else "failed"
    reconciliation = "not_applicable"
    difference = None
    if plan.reported_value is not None:
        quantum = reported_quantum(plan)
        reconciliation = (
            "match"
            if abs(plan.reported_value - published_recomputed) <= quantum
            else "mismatch"
        )
        difference = plan.reported_value - published_recomputed

    scored_frame = frame.copy()
    per_row: list[float | None] = [None] * len(frame)
    for record, score in zip(records, scores.tolist()):
        if not pd.isna(score):
            for row in record["rows"]:
                per_row[row] = float(score)
    scored_frame["main_metric"] = per_row
    scored_units = int(scored_mask.sum())
    coverage = {
        "total_units": len(records),
        "scored_units": scored_units,
        "excluded_units": len(records) - scored_units,
        "weight_sum": float(total_weight),
    }
    km = {
        "report_version": "laim-km.v2",
        "status": "computed",
        "value_source": source,
        "recomputed_value": float(published_recomputed),
        "coverage": coverage,
        "reconciliation": {
            "status": reconciliation,
            "difference": float(difference) if difference is not None else None,
        },
        "evidence": {key: list(value) for key, value in plan.evidence.items()},
        "threshold_verdict": verdict,
        "main_metric": {
            "name": plan.metric_name,
            "value": float(final_value),
            "recomputed_value": float(published_recomputed),
            "value_source": source,
            "n_units": scored_units,
            "units_dropped_nan_score": len(records) - scored_units,
            "evaluation_unit": plan.evaluation_unit,
            "scoring_method": plan.method,
            "formula": formula.text,
            "aggregation": plan.reducer,
            "weighted": weighted,
            "missing_policy": plan.missing_policy,
            "majority_denominator": plan.majority_denominator,
            "scale": plan.scale,
            "precision": plan.precision,
            "threshold": float(plan.threshold) if plan.threshold is not None else None,
            "comparator": plan.comparator,
            "threshold_verdict": verdict,
            "reconciliation_status": reconciliation,
        },
    }
    return scored_frame, km
