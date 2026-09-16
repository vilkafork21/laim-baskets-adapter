"""КМ выборки: компоненты/счётчики накапливаются ДО нелинейной агрегации.

Это не построчный score: результаты нельзя усреднять между строками или
батчами. Для воспроизведения сохраняются достаточные статистики.
"""

from __future__ import annotations

from decimal import Decimal

from ..errors import NotEvaluableError
from ..models import MeasurementPlan
from ..transform.values import normalize_key

METHODS = frozenset({"harmonic_mean_of_means", "classification_f1"})
_COMPONENTS = ("precision_component", "recall_component")
_ZERO = Decimal(0)


def validate_options(proposal: dict) -> dict:
    """Новый метод требует явных параметров; старые методы их не игнорируют."""
    method = proposal["method"]
    options = dict(proposal.get("metric_options", {}))
    if method not in METHODS:
        if options:
            raise NotEvaluableError("metric_options допустимы только для неаддитивной КМ")
        return options
    expected = {"zero_division"}
    if method == "classification_f1":
        expected |= {"average", "labels", "positive_label"}
    if set(options) != expected:
        raise NotEvaluableError(
            "Этап S: параметры неаддитивной КМ должны быть заданы явно",
            expected=sorted(expected),
            observed=sorted(options),
            repair_hint="Укажите metric_options по методике; не выбирайте вариант по числу baseline",
        )
    if proposal["scale"] not in {"ratio", "percent"}:
        raise NotEvaluableError(
            "F1 требует ratio/percent; приведите компоненты к 0..1 явным value_map"
        )
    if method == "harmonic_mean_of_means":
        return options
    if proposal["missing_policy"] not in {"fail", "exclude_unit"}:
        raise NotEvaluableError(
            "classification_f1: отсутствующий класс нельзя заменить нулём; "
            "допустимы missing_policy=fail или exclude_unit"
        )
    average, positive = options["average"], options["positive_label"]
    if average == "binary":
        if positive is None or not normalize_key(positive) or options["labels"] is not None:
            raise NotEvaluableError("binary требует positive_label и labels=null")
        options["positive_label"] = normalize_key(positive)
    elif positive is not None:
        raise NotEvaluableError("positive_label допустим только при average=binary")
    if options["labels"] is not None:
        labels = [normalize_key(label) for label in options["labels"]]
        if not labels or any(not label for label in labels) or len(labels) != len(set(labels)):
            raise NotEvaluableError("labels должны быть непустыми и уникальными после нормализации")
        options["labels"] = labels
    return options


def _divide(numerator: Decimal, denominator: Decimal, zero_division, context: str) -> Decimal:
    if denominator:
        return numerator / denominator
    if zero_division == "fail":
        raise NotEvaluableError(
            f"Этап S: нулевой знаменатель {context}; metric_options.zero_division=fail. "
            "Уточните методику либо задайте явную политику 0/1.",
        )
    return Decimal(zero_division)


def _weight(record: dict, plan: MeasurementPlan) -> Decimal:
    weight = record["weight"] if plan.reducer == "frequency_weighted_mean" else Decimal(1)
    if not weight.is_finite() or weight <= 0:
        raise NotEvaluableError("Вес оцененной единицы должен быть конечным и положительным")
    return weight


def _coverage(total: int, count: int, weight: Decimal) -> dict:
    return {
        "total_units": total,
        "scored_units": count,
        "excluded_units": total - count,
        "weight_sum": float(weight),
    }


def _harmonic(records: list[dict], plan: MeasurementPlan) -> tuple[Decimal, dict, dict]:
    columns = {source["role"]: source["column_id"] for source in plan.sources}
    totals = {role: _ZERO for role in _COMPONENTS}
    weights = dict(totals)
    counts = {role: 0 for role in _COMPONENTS}
    included = complete = partial = 0
    total_weight = _ZERO
    for record in records:
        values = {role: record["values"][columns[role]] for role in _COMPONENTS}
        missing = any(value is None for value in values.values())
        if missing and plan.missing_policy == "fail":
            raise NotEvaluableError("Отсутствует компонент F1, а missing_policy=fail")
        if missing and plan.missing_policy == "exclude_unit":
            continue
        if plan.missing_policy == "zero":
            values = {role: value if value is not None else _ZERO for role, value in values.items()}
        if all(value is None for value in values.values()):
            continue
        weight = _weight(record, plan)
        included += 1
        total_weight += weight
        complete += int(not missing)
        partial += int(missing)
        for role, value in values.items():
            if value is not None:
                if not _ZERO <= value <= 1:
                    raise NotEvaluableError("Нормализованный компонент F1 должен быть в 0..1")
                totals[role] += value * weight
                weights[role] += weight
                counts[role] += 1
    if any(not weights[role] for role in _COMPONENTS):
        raise NotEvaluableError(
            "Этап S: нет оценок хотя бы одного компонента F1; "
            "проверьте источники и политику пропусков"
        )
    means = {role: totals[role] / weights[role] for role in _COMPONENTS}
    precision, recall = (means[role] for role in _COMPONENTS)
    value = _divide(
        2 * precision * recall,
        precision + recall,
        plan.metric_options["zero_division"],
        "гармонического среднего",
    )
    stats = {
        "method": plan.method,
        "formula": "2 * mean(precision) * mean(recall) / (mean(precision) + mean(recall))",
        "complete_units": complete,
        "partial_units": partial,
        "components": {
            role: {
                "weighted_sum": str(totals[role]),
                "weight_sum": str(weights[role]),
                "observed_units": counts[role],
                "mean": str(means[role]),
            }
            for role in _COMPONENTS
        },
    }
    return value, _coverage(len(records), included, total_weight), stats


def _classification(records: list[dict], plan: MeasurementPlan) -> tuple[Decimal, dict, dict]:
    columns = {source["role"]: source["column_id"] for source in plan.sources}
    # Один проход по единицам; не строим N x число_классов и не сохраняем тексты строк.
    true, predicted, hits = {}, {}, {}
    included = 0
    total_weight = _ZERO
    for record in records:
        pred, target = (record["values"][columns[role]] for role in ("prediction", "target"))
        if pred is None or target is None:
            if plan.missing_policy == "fail":
                raise NotEvaluableError("Отсутствует prediction/target при missing_policy=fail")
            continue
        weight = _weight(record, plan)
        included += 1
        total_weight += weight
        true[target] = true.get(target, _ZERO) + weight
        predicted[pred] = predicted.get(pred, _ZERO) + weight
        if target == pred:
            hits[target] = hits.get(target, _ZERO) + weight
    if not included:
        raise NotEvaluableError("Ни одной полной пары prediction/target для F1")
    options = plan.metric_options
    observed = set(true) | set(predicted)
    average = options["average"]
    if average == "binary":
        positive = options["positive_label"]
        if len(observed) > 2 or (len(observed) == 2 and positive not in observed):
            raise NotEvaluableError(
                "binary несовместим с классами выборки или positive_label",
                observed=sorted(observed),
                positive_label=positive,
            )
        labels = [positive]
    else:
        labels = options["labels"] if options["labels"] is not None else sorted(observed)
    counters = []
    for label in labels:
        tp = hits.get(label, _ZERO)
        support = true.get(label, _ZERO)
        fp, fn = predicted.get(label, _ZERO) - tp, support - tp
        denominator = 2 * tp + fp + fn
        counters.append((label, tp, fp, fn, support, denominator))
    zero = options["zero_division"]
    if average in {"micro", "binary"}:
        value = _divide(
            sum((2 * c[1] for c in counters), _ZERO),
            sum((c[5] for c in counters), _ZERO),
            zero,
            "F1",
        )
    else:
        supports = [c[4] for c in counters]
        class_weights = (
            supports if average == "weighted" and sum(supports) else [Decimal(1)] * len(labels)
        )
        # Класс с нулевой поддержкой не влияет на weighted, но входит в macro.
        value = sum(
            (
                _divide(2 * tp, denominator, zero, f"F1 класса {label!r}") * weight
                for (label, tp, _fp, _fn, _support, denominator), weight in zip(
                    counters, class_weights
                )
                if weight
            ),
            _ZERO,
        ) / sum(class_weights)
    stats = {
        "method": plan.method,
        "average": average,
        "labels": labels,
        "observed_labels": sorted(observed),
        "per_class": [
            {
                "label": label,
                "tp": str(tp),
                "fp": str(fp),
                "fn": str(fn),
                "support": str(support),
                "denominator": str(denominator),
                "f1": str(2 * tp / denominator)
                if denominator
                else (None if zero == "fail" else str(Decimal(zero))),
            }
            for label, tp, fp, fn, support, denominator in counters
        ],
    }
    return value, _coverage(len(records), included, total_weight), stats


def aggregate(records: list[dict], plan: MeasurementPlan) -> tuple[Decimal, dict, dict]:
    """Итог в долях, покрытие, достаточные статистики с точными Decimal-строками."""
    if plan.method == "harmonic_mean_of_means":
        return _harmonic(records, plan)
    if plan.method == "classification_f1":
        return _classification(records, plan)
    raise NotEvaluableError("Неизвестный неаддитивный метод", method=plan.method)
