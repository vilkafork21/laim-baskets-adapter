"""Чистый Decimal-вычислитель по валидированному MeasurementPlan."""

from __future__ import annotations

import logging
from decimal import Decimal

from ..errors import MeasurementPlanError, NotEvaluableError
from .resolve import decimal_value, reported_quantum
from ..models import MeasurementPlan, ResolvedLayout
from ..transform.values import blank as _blank, normalize_key

logger = logging.getLogger(__name__)


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
        def base(value: object) -> Decimal | None:
            try:
                return _numeric(value)
            except MeasurementPlanError:
                # Нечисловой текст в колонке оценки — пропуск в данных, а не
                # поломка плана: решение принимает missing_policy, как для пустой
                # ячейки. Иначе одна ячейка убивает разбор всей корзины.
                logger.warning(
                    "Колонка %s: нечисловое значение %r трактуется как пропуск оценки",
                    source["column_id"], str(value)[:80],
                )
                return None
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


_PERCENT_DOMAIN_MAX = Decimal(100)


def _percent_domain(values: list[object], column_id: str) -> list[object] | None:
    """Оценки в процентных пунктах (0-100 без знака %) при шкале доли
    (percent или ratio) приводятся к долям; иначе main_metric ушёл бы
    потребителям на 0-100, а пересчёт КМ — умноженным на 100 (LAIM-0189).
    Шкала raw (оценка 0-2 и подобные) не трогается."""
    present = [value for value in values if value is not None]
    if not present or max(present) <= 1:
        return None
    if max(present) > _PERCENT_DOMAIN_MAX:
        raise NotEvaluableError(
            "Оценки колонки выходят за домен percent (0-100)",
            column_id=column_id, max_value=str(max(present)),
        )
    return [None if value is None else value / _PERCENT_DOMAIN_MAX for value in values]


def source_values(frame, layout: ResolvedLayout, plan: MeasurementPlan) -> dict[str, list[object]]:
    return {column_id: values for column_id, values, _ in _sources(frame, layout, plan)}


def _sources(frame, layout: ResolvedLayout, plan: MeasurementPlan):
    for source in plan.sources:
        column_id = source["column_id"]
        normalizer = _normalizer(source)
        values = [normalizer(value) for value in frame[layout.column_names[column_id]].tolist()]
        normalized = None
        if plan.scale in ("percent", "ratio") and source["normalization"] == "numeric":
            normalized = _percent_domain(values, column_id)
        yield column_id, normalized if normalized is not None else values, normalized is not None


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
    sources = list(_sources(frame, layout, plan))
    values = {column_id: column_values for column_id, column_values, _ in sources}
    percent_columns = [column_id for column_id, _, normalized in sources if normalized]
    if percent_columns:
        logger.warning(
            "Колонки %s несут оценки в процентных пунктах при шкале %s — "
            "приведены к долям делением на 100", percent_columns, plan.scale,
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
    recomputed = sum(
        score * weight for (_record, score), weight in zip(scored, weights)
    ) / total_weight
    published_recomputed = _published_scale(recomputed, plan.scale)
    # КМ — только заявленная в отчёте о валидации: без неё value и вердикт
    # пусты, пересчёт остаётся информационным полем.
    final_value = plan.reported_value
    verdict = None
    if plan.threshold is not None and final_value is not None:
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
        logger.info(
            "Сверка КМ: пересчёт %s, заявлено в отчёте %s, допуск %s -> %s (расхождение %s)",
            published_recomputed, plan.reported_value, quantum, reconciliation, difference,
        )
    else:
        logger.info("Отчёт о валидации не объявил КМ: value пуст, информационный "
                    "пересчёт %s", published_recomputed)
    logger.info(
        "Оценено единиц %d из %d, суммарный вес %s, вердикт по порогу %s",
        len(scored), len(records), total_weight, verdict,
    )

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
    # Внутренняя сводка прогона: контракт монитора собирает main.py из плана,
    # а журнал прогона — pipeline. Здесь только то, что они реально читают.
    km = {
        "recomputed_value": float(published_recomputed),
        "coverage": coverage,
        "reconciliation": {
            "status": reconciliation,
            "difference": float(difference) if difference is not None else None,
        },
        "threshold_verdict": verdict,
        "percent_domain_columns": percent_columns,
        "main_metric": {
            "name": plan.metric_name,
            "value": float(final_value) if final_value is not None else None,
            "recomputed_value": float(published_recomputed),
            "scale": plan.scale,
        },
    }
    return scored_frame, km
