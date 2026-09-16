"""Чистый Decimal-вычислитель по валидированному MeasurementPlan."""

from __future__ import annotations

import logging
from decimal import Decimal

from ..errors import MeasurementPlanError, NotEvaluableError
from ..models import MeasurementPlan, ResolvedLayout
from ..transform.values import blank as _blank
from ..transform.values import normalize_key
from .resolve import decimal_value, reported_quantum

logger = logging.getLogger(__name__)


def _numeric(value: object, scale: str) -> Decimal | None:
    if _blank(value):
        return None
    if isinstance(value, bool):
        raise NotEvaluableError("Boolean нельзя использовать как неявный numeric score")
    text = str(value).strip()
    percent = text.endswith("%")
    result = decimal_value(text.rstrip("%").strip())
    if percent and scale == "raw":
        raise NotEvaluableError("Процентная ячейка несовместима со шкалой raw")
    return result / 100 if percent or scale == "percent" else result


def _normalizer(source: dict[str, object], scale: str):
    normalization = source["normalization"]
    if normalization == "label":
        return lambda value: None if _blank(value) else normalize_key(value)
    if normalization == "numeric":

        def base(value: object) -> Decimal | None:
            try:
                return _numeric(value, scale)
            except MeasurementPlanError as exc:
                raise NotEvaluableError(
                    "Нечисловой score при normalization=numeric; задайте явный value_map",
                    column_id=source["column_id"],
                    value=str(value)[:120],
                    repair_hint="Используйте все unique_values инвентаря; не исключайте текст как пропуск",
                ) from exc
    else:
        lookup = {normalize_key(key): decimal_value(value) for key, value in normalization.items()}

        def base(value: object) -> Decimal | None:
            if _blank(value):
                return None
            key = normalize_key(value)
            if key not in lookup:
                raise NotEvaluableError(
                    "value_map не покрывает фактическое значение", value=str(value)[:80]
                )
            return lookup[key] / 100 if scale == "percent" else lookup[key]

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
    return {column_id: values for column_id, values, _ in _sources(frame, layout, plan)}


def _sources(frame, layout: ResolvedLayout, plan: MeasurementPlan):
    for source in plan.sources:
        column_id = source["column_id"]
        normalizer = _normalizer(source, plan.scale)
        raw = frame[layout.column_names[column_id]].tolist()
        missing_tokens = {normalize_key(value) for value in plan.missing_values.get(column_id, ())}
        values = [
            None if normalize_key(value) in missing_tokens else normalizer(value) for value in raw
        ]
        numeric = source["normalization"] != "label"
        if numeric and plan.scale in ("ratio", "percent"):
            invalid = [
                str(value)
                for value in values
                if value is not None and not Decimal(0) <= value <= Decimal(1)
            ]
            if invalid:
                raise NotEvaluableError(
                    "Оценки не соответствуют явной шкале плана S",
                    column_id=column_id,
                    scale=plan.scale,
                    invalid=invalid[:10],
                    repair_hint="Уточните scale: ratio 0..1, percent 0..100, raw для иной шкалы",
                )
        percent = numeric and (
            plan.scale == "percent"
            or any(isinstance(value, str) and value.rstrip().endswith("%") for value in raw)
        )
        yield column_id, values, percent


def _unit_records(
    frame, values: dict[str, list[object]], plan: MeasurementPlan
) -> list[dict[str, object]]:
    if plan.evaluation_unit == "turn":
        return [
            {
                "values": {
                    column: column_values[index] for column, column_values in values.items()
                },
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
            present = {
                column_values[index] for index in indexes if column_values[index] is not None
            }
            if len(present) > 1:
                raise NotEvaluableError(
                    "Источник КМ не константен внутри dialogue",
                    group=str(group),
                    column_id=column,
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


def evaluate(
    frame, layout: ResolvedLayout, plan: MeasurementPlan
) -> tuple[object, dict[str, object]]:
    sources = list(_sources(frame, layout, plan))
    values = {column_id: column_values for column_id, column_values, _ in sources}
    percent_columns = [column_id for column_id, _, normalized in sources if normalized]
    if percent_columns:
        logger.warning(
            "Колонки %s несут оценки в процентных пунктах при шкале %s — "
            "приведены к долям делением на 100",
            percent_columns,
            plan.scale,
        )
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
    recomputed = (
        sum(score * weight for (_record, score), weight in zip(scored, weights)) / total_weight
    )
    published_recomputed = _published_scale(recomputed, plan.scale)
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
        "recomputed_value": float(published_recomputed),
        "recomputed_exact": str(published_recomputed),
        "coverage": coverage,
        "percent_domain_columns": percent_columns,
    }
    return scored_frame, reconcile(km, plan)


def reconcile(km: dict, plan: MeasurementPlan) -> dict:
    """Присоединить K к уже рассчитанному S, не трогая оценки и выборку."""
    recomputed = Decimal(km.get("recomputed_exact", str(km["recomputed_value"])))
    declared = plan.reported_value
    difference = None if declared is None else declared - recomputed
    status = (
        "not_applicable"
        if difference is None
        else ("match" if abs(difference) <= reported_quantum(plan) else "mismatch")
    )
    verdict = None
    if declared is not None and plan.threshold is not None:
        passed = (
            declared >= plan.threshold if plan.comparator == ">=" else declared <= plan.threshold
        )
        verdict = "passed" if passed else "failed"
    return {
        **km,
        "reconciliation": {
            "status": status,
            "difference": None if difference is None else float(difference),
        },
        "threshold_verdict": verdict,
        "main_metric": {
            "name": plan.metric_name,
            "value": None if declared is None else float(declared),
            "recomputed_value": float(recomputed),
            "scale": plan.scale,
        },
    }
