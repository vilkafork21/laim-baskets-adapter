"""Независимая контрольная арифметика проверенного плана, не вызов движка ноды.

Читает числовые ячейки напрямую, группирует по проверенному якорю, считает
доли/средние. Не подтверждает правильность выбора методики аудитором или LLM.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import openpyxl
from openpyxl.utils import column_index_from_string


def label(value):
    if value is None or not str(value).strip():
        return None
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def number(value, source, scale):
    if label(value) is None:
        return None
    normalization = source["normalization"]
    if normalization == "label":
        return label(value)
    if isinstance(normalization, dict):
        value = {label(k): v for k, v in normalization.items()}[label(value)]
    text = str(value).strip()
    percent = text.endswith("%")
    result = Decimal(re.sub(r"\s+", "", text.rstrip("%")).replace(",", "."))
    if percent or scale == "percent":
        result /= 100
    return 1 - result if source["polarity"] == "inverted" else result


def nonlinear(samples, plan):
    """Независимый Fraction-пересчёт: не импортирует код агрегатора ноды."""
    method = plan["method"]
    options = plan["metric_options"]
    roles = [source["role"] for source in plan["sources"]]
    policy = plan["missing_policy"]
    kept = []
    for values, weight in samples:
        missing = any(v is None for v in values)
        if missing and policy == "fail":
            raise ValueError("Пропуск при policy=fail")
        if missing and policy == "exclude_unit":
            continue
        if missing and policy == "zero":
            values = [0 if v is None else v for v in values]
        if all(v is None for v in values):
            continue
        kept.append((dict(zip(roles, values)), Fraction(weight)))

    def divide(a, b):
        if b:
            return a / b
        if options["zero_division"] == "fail":
            raise ValueError("Неопределённая F1 по заданной политике")
        return Fraction(options["zero_division"])

    if method == "harmonic_mean_of_means":
        stats = {}
        means = []
        for role in ("precision_component", "recall_component"):
            pairs = [(Fraction(v[role]), w) for v, w in kept if v[role] is not None]
            total = sum((v * w for v, w in pairs), Fraction(0))
            den = sum((w for v, w in pairs), Fraction(0))
            if not den:
                raise ValueError("Компонент отсутствует целиком")
            means.append(total / den)
            stats[role] = {"sum": str(total), "weight": str(den), "units": len(pairs)}
        a, b = means
        value = divide(2 * a * b, a + b)
    else:
        labels = options["labels"]
        if options["average"] == "binary":
            labels = [label(options["positive_label"])]
        elif labels is None:
            labels = sorted({v[role] for v, _w in kept for role in ("target", "prediction")})
        else:
            labels = [label(v) for v in labels]
        counts = []
        for cls in labels:
            tp = sum((w for v, w in kept if v["target"] == v["prediction"] == cls), Fraction(0))
            fp = sum((w for v, w in kept if v["prediction"] == cls != v["target"]), Fraction(0))
            fn = sum((w for v, w in kept if v["target"] == cls != v["prediction"]), Fraction(0))
            counts.append((tp, fp, fn))
        if options["average"] in {"binary", "micro"}:
            value = divide(
                sum(2 * tp for tp, fp, fn in counts), sum(2 * tp + fp + fn for tp, fp, fn in counts)
            )
        else:
            weights = (
                [tp + fn for tp, fp, fn in counts]
                if options["average"] == "weighted"
                else [1] * len(counts)
            )
            weights = weights if sum(weights) else [1] * len(counts)
            value = sum(
                divide(2 * tp, 2 * tp + fp + fn) * w
                for (tp, fp, fn), w in zip(counts, weights)
                if w
            ) / sum(weights)
        stats = {
            "counts": [
                {"label": cls, "tp": str(tp), "fp": str(fp), "fn": str(fn)}
                for cls, (tp, fp, fn) in zip(labels, counts)
            ]
        }
    return value, len(kept), stats


def verify(case, root, result):
    proposals = case.get("reviewed_proposals")
    if not proposals:
        return {"case_id": case["case_id"], "status": "not_applicable_incomplete"}
    layout, plan = proposals["layout"], proposals["metric"]
    method = plan["method"]
    if method not in {
        "identity",
        "accuracy",
        "majority",
        "mean_criteria",
        "all_criteria",
        "all_assessors",
        "harmonic_mean_of_means",
        "classification_f1",
    }:
        return {"case_id": case["case_id"], "status": "unsupported_method", "method": method}
    path = root / case["inputs"]["test_set"]
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(book[case["sheet_name"]].iter_rows(values_only=True))
    finally:
        book.close()
    query_column = column_index_from_string(layout["roles"]["input_query"]) - 1
    sources = plan["sources"]

    def cell(row, column):
        return row[column_index_from_string(column) - 1]

    units = defaultdict(list)
    group = None
    kind = layout["grouping"]["kind"]
    dialogue = plan.get("assessment_mode") == "dialogue" or kind == "blob_row"
    for index, row in enumerate(rows[max(layout["header_rows"]) :], max(layout["header_rows"]) + 1):
        if label(row[query_column]) is None:
            continue
        if dialogue and kind != "blob_row":
            group = cell(row, layout["grouping"]["column"]) or group
            key = group
        else:
            key = index
        units[key].append(row)
    numerator, denominator = Decimal(0), Decimal(0)
    scored = 0
    samples = []
    for records in units.values():
        vals = []
        for source in sources:
            raw = next(
                (
                    cell(row, source["column_id"])
                    for row in records
                    if label(cell(row, source["column_id"])) is not None
                ),
                None,
            )
            if label(raw) in {
                label(v) for v in plan.get("missing_values", {}).get(source["column_id"], [])
            }:
                raw = None
            vals.append(number(raw, source, plan["scale"]))
        if method in {"harmonic_mean_of_means", "classification_f1"}:
            weight = (
                Decimal(str(cell(records[0], layout["weight_column"])))
                if plan["reducer"] == "frequency_weighted_mean"
                else Decimal(1)
            )
            samples.append((vals, weight))
            continue
        if any(v is None for v in vals):
            policy = plan["missing_policy"]
            if policy == "fail":
                raise ValueError("Пропуск при policy=fail")
            if policy == "zero":
                vals = [Decimal(0) if v is None else v for v in vals]
            elif policy == "exclude_value" and method in {"mean_criteria", "all_criteria"}:
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
            else:
                continue
        if method == "identity":
            score = vals[0]
        elif method == "accuracy":
            score = Decimal(int(vals[0] == vals[1]))
        elif method == "majority":
            score = Decimal(int(sum(vals) > Decimal(len(vals)) / 2))
        elif method in {"all_criteria", "all_assessors"}:
            score = min(vals)
        else:
            score = sum(vals) / len(vals)
        weight = (
            Decimal(str(cell(records[0], layout["weight_column"])))
            if plan["reducer"] == "frequency_weighted_mean"
            else Decimal(1)
        )
        numerator += score * weight
        denominator += weight
        scored += 1
    extra = {}
    if method in {"harmonic_mean_of_means", "classification_f1"}:
        exact, scored, statistics = nonlinear(samples, plan)
        numerator, denominator = Decimal(exact.numerator), Decimal(exact.denominator)
        extra = {"method": method, "independent_statistics": statistics}
    value = numerator / denominator
    if plan["scale"] == "percent":
        value *= 100
    observed = result.get("recomputed")
    return {
        "case_id": case["case_id"],
        "status": "checked",
        **extra,
        "units": len(units),
        "scored_units": scored,
        "numerator": str(numerator),
        "denominator": str(denominator),
        "independent_value": str(value),
        "node_value": observed,
        "matches_node": observed is not None
        and abs(value - Decimal(str(observed))) < Decimal("1e-12"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "root", "results", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text())["cases"]
    results = {r["case_id"]: r for r in json.loads(args.results.read_text())}
    checked = [verify(case, args.root, results[case["case_id"]]) for case in cases]
    args.out.write_text(json.dumps(checked, ensure_ascii=False, indent=2) + "\n")
    failed = [c["case_id"] for c in checked if c.get("matches_node") is False]
    print(json.dumps({"checked": sum(c["status"] == "checked" for c in checked), "failed": failed}))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
