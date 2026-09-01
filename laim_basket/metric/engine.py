"""Чистый Decimal-вычислитель по валидированному MeasurementPlan."""

from __future__ import annotations

from decimal import Decimal

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


def _missing_score(plan: MeasurementPlan, reason: str) -> Decimal | None:
    if plan.missing_policy == "fail":
        raise NotEvaluableError(reason)
    if plan.missing_policy in {"exclude_unit", "exclude_value"}:
        return None
    return Decimal(0)


def _present_values(
    values: list[object], plan: MeasurementPlan, reason: str
) -> list[Decimal] | None:
    """Применить missing_policy к значениям единицы: None — единица не оценивается."""
    if plan.missing_policy == "exclude_value":
        values = [value for value in values if value is not None]
        return values or None
    if any(value is None for value in values):
        replacement = _missing_score(plan, reason)
        if replacement is None:
            return None
        values = [replacement if value is None else value for value in values]
    return values


def _min_binary(values: list[Decimal], method: str) -> Decimal:
    if any(value not in (Decimal(0), Decimal(1)) for value in values):
        raise NotEvaluableError(f"{method} принимает только нормализованные 0/1")
    return min(values)


def _score(record: dict[str, object], plan: MeasurementPlan) -> Decimal | None:
    by_role: dict[str, list[object]] = {}
    for source in plan.sources:
        by_role.setdefault(source["role"], []).append(record["values"][source["column_id"]])
    if plan.method == "identity":
        value = by_role["final_score"][0]
        return _missing_score(plan, "Отсутствует final score") if value is None else value
    if plan.method == "accuracy":
        prediction, target = by_role["prediction"][0], by_role["target"][0]
        if prediction is None or target is None:
            return _missing_score(plan, "Отсутствует prediction или target")
        return Decimal(int(prediction == target))
    if plan.method in ("mean_criteria", "all_criteria"):
        values = _present_values(by_role["criterion"], plan, "Отсутствует criterion score")
        if values is None:
            return None
        if plan.method == "mean_criteria":
            return sum(values, Decimal(0)) / len(values)
        return _min_binary(values, "all_criteria")
    votes = by_role["assessor_vote"]
    if plan.method == "all_assessors":
        values = _present_values(votes, plan, "Отсутствует голос assessor")
        return None if values is None else _min_binary(values, "all_assessors")
    present = [vote for vote in votes if vote is not None]
    if any(vote not in (Decimal(0), Decimal(1)) for vote in present):
        raise NotEvaluableError("majority принимает только бинарные голоса")
    if not present:
        return _missing_score(plan, "Нет голосов assessor")
    denominator = len(votes) if plan.majority_denominator == "declared" else len(present)
    positives = sum(present, Decimal(0))
    if positives * 2 == denominator:
        return _missing_score(plan, "Majority завершился ничьей")
    return Decimal(int(positives * 2 > denominator))


def _published_scale(value: Decimal, scale: str) -> Decimal:
    return value * 100 if scale == "percent" else value


def evaluate(frame, layout: ResolvedLayout, plan: MeasurementPlan) -> tuple[object, dict[str, object]]:
    values = source_values(frame, layout, plan)
    records = _unit_records(frame, values, plan)
    scores = [_score(record, plan) for record in records]
    scored = [(record, score) for record, score in zip(records, scores) if score is not None]
    if not scored:
        raise NotEvaluableError("Ни одной оцененной единицы")
    weighted = plan.reducer == "frequency_weighted_mean"
    weights = [record["weight"] if weighted else Decimal(1) for record, _score_value in scored]
    if any(weight <= 0 for weight in weights):
        raise NotEvaluableError("Вес должен быть положительным")
    total_weight = sum(weights, Decimal(0))
    recomputed = sum(
        score * weight for (_record, score), weight in zip(scored, weights)
    ) / total_weight
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
    for record, score in zip(records, scores):
        if score is not None:
            for row in record["rows"]:
                per_row[row] = float(score)
    scored_frame["main_metric"] = per_row
    coverage = {
        "total_units": len(records),
        "scored_units": len(scored),
        "excluded_units": len(records) - len(scored),
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
            "n_units": len(scored),
            "units_dropped_nan_score": len(records) - len(scored),
            "evaluation_unit": plan.evaluation_unit,
            "scoring_method": plan.method,
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
