"""Построчный score единиц и агрегация КМ формулой контракта."""

from __future__ import annotations

import pandas as pd

from . import formula
from .contract import WEIGHT, source_name, validate_monitoring_metric
from .errors import MonitoringContractError
from .units import unitize
from .values import blank


def _normalized_column(units: pd.DataFrame, source: dict) -> pd.Series:
    """Метки — текст, числа — float; пустое — NaN."""
    raw = units[source["source_id"]]
    if source["normalization"] == "label":
        return raw.map(lambda v: None if blank(v) else str(v)).astype(object)
    return pd.to_numeric(raw, errors="coerce").astype("float64")


def formula_columns(units: pd.DataFrame, contract: dict) -> dict[str, pd.Series]:
    """Входы формулы, доступные в units; отсутствующие колонки просто не попадают."""
    columns = {
        source_name(source): _normalized_column(units, source)
        for source in contract["scoring"]["sources"]
        if source["source_id"] in units
    }
    if "input_query_count" in units:
        columns[WEIGHT] = pd.to_numeric(units["input_query_count"], errors="coerce").astype("float64")
    return columns


def _check_blanks(columns: dict[str, pd.Series], names: tuple[str, ...], policy: str) -> None:
    if policy != "fail":
        return
    for name in names:
        if name in columns and columns[name].isna().any():
            raise MonitoringContractError(f"missing_policy=fail: вход {name!r} содержит пропуски")


def unit_scores(units: pd.DataFrame, contract: dict) -> pd.Series:
    """Построчный score единиц: построчная часть формулы mean(E)/wmean(E, w).

    Для формул без построчной части (macro-F1 и т.п.) — совпадение prediction
    с target, если обе роли есть, иначе NaN: такой метрике построчный score
    не нужен, он служит только дрифт-тестам и примерам в отчёте.
    """
    parsed = formula.parse(contract["formula"])
    columns = formula_columns(units, contract)
    unit = parsed.unit_expression()
    if unit is None:
        by_role = {source["role"]: source_name(source) for source in contract["scoring"]["sources"]}
        if "prediction" in by_role and "target" in by_role:
            unit = formula.parse(f"{by_role['prediction']} == {by_role['target']}")
        else:
            return pd.Series(float("nan"), index=units.index, dtype="float64")
    missing = [name for name in unit.inputs if name not in columns]
    if missing:
        raise MonitoringContractError(f"Нет входов формулы в единицах оценки: {missing}")
    _check_blanks(columns, unit.inputs, contract["scoring"]["missing_policy"])
    try:
        return unit.evaluate_rows(columns).astype("float64")
    except formula.FormulaError as exc:
        raise MonitoringContractError(f"Формула КМ: {exc}") from exc


def score_units(units: pd.DataFrame, payload: dict) -> pd.Series:
    return unit_scores(units, validate_monitoring_metric(payload))


def aggregate_main_metric(frame: pd.DataFrame, payload: dict) -> dict[str, object]:
    """КМ по единицам оценки той же формулой, что baseline на корзине."""
    contract = validate_monitoring_metric(payload)
    units = unitize(frame, contract)
    policy = contract["scoring"]["missing_policy"]
    parsed = formula.parse(contract["formula"])
    columns = formula_columns(units, contract)
    missing = [name for name in parsed.inputs if name not in columns]
    if missing:
        raise MonitoringContractError(
            f"В данных нет входов формулы {missing}: разметка судьи должна лежать в "
            "колонках контракта, ответ агента — в UMR"
        )
    _check_blanks(columns, parsed.inputs, policy)
    try:
        value = parsed.evaluate(columns)
        unit = parsed.unit_expression()
        scored = (
            unit.evaluate_rows(columns).notna() if unit is not None
            else pd.concat([columns[name] for name in parsed.inputs], axis=1).notna().all(axis=1)
        )
    except formula.FormulaError as exc:
        raise MonitoringContractError(f"Формула КМ: {exc}") from exc
    if pd.isna(value) or not scored.any():
        raise MonitoringContractError("Нет оцененных единиц")
    weighted = contract["aggregation"]["method"] == "frequency_weighted_mean"
    weights = columns[WEIGHT] if weighted else pd.Series(1.0, index=units.index)
    return {
        "name": contract["name"],
        "value": float(value),
        "formula": parsed.text,
        "total_units": int(len(units)),
        "scored_units": int(scored.sum()),
        "excluded_units": int((~scored).sum()),
        "weight_sum": float(weights[scored].sum()),
    }


def broadcast_scores(frame: pd.DataFrame, units: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    if len(units) != len(scores):
        raise MonitoringContractError("Число scores не совпадает с числом units")
    result = frame.copy()
    result["main_metric"] = None
    result["assessment_unit_id"] = None
    score_column = result.columns.get_loc("main_metric")
    unit_column = result.columns.get_loc("assessment_unit_id")
    for unit_id, positions, score in zip(
        units["_unit_id"], units["_row_positions"], scores.tolist()
    ):
        for position in positions:
            result.iat[position, score_column] = score
            result.iat[position, unit_column] = unit_id
    result["main_metric"] = pd.to_numeric(result["main_metric"], errors="coerce")
    return result
