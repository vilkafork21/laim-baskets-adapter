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
    value = numerator / denominator
    if plan["scale"] == "percent":
        value *= 100
    observed = result.get("recomputed")
    return {
        "case_id": case["case_id"],
        "status": "checked",
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
