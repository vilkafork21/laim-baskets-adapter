"""Расчёт КМ формулой контракта: одно число на набор единиц и построчный score."""

from __future__ import annotations

import pandas as pd

from . import formula
from .contract import WEIGHT, agent_inputs, judged_inputs, uses_weight, validate_monitoring_metric
from .errors import MonitoringContractError
from .units import unitize


def _coerce(series: pd.Series) -> pd.Series:
    """Числа — float64, всё остальное — метки (object)."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        return numeric.astype("float64")
    return series.astype(object).where(series.notna(), None)


def formula_columns(units: pd.DataFrame, contract: dict) -> dict[str, pd.Series]:
    """Входы формулы, которые есть в единицах оценки."""
    columns = {
        item["name"]: _coerce(units[item["name"]])
        for item in contract["inputs"]
        if item["name"] in units
    }
    if "input_query_count" in units:
        columns[WEIGHT] = pd.to_numeric(units["input_query_count"], errors="coerce").astype("float64")
    return columns


def evaluate_formula(units: pd.DataFrame, contract: dict) -> dict[str, object]:
    """Значение формулы по единицам и покрытие (сколько единиц вошло в расчёт)."""
    parsed = formula.parse(contract["formula"])
    columns = formula_columns(units, contract)
    missing = [name for name in parsed.inputs if name not in columns]
    if missing:
        raise MonitoringContractError(
            f"В данных нет входов формулы {missing}: разметка судьи должна лежать в "
            "колонках контракта, ответ агента — в UMR"
        )
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
    weights = columns[WEIGHT] if uses_weight(contract) else pd.Series(1.0, index=units.index)
    return {
        "value": float(value),
        "formula": parsed.text,
        "total_units": int(len(units)),
        "scored_units": int(scored.sum()),
        "excluded_units": int((~scored).sum()),
        "weight_sum": float(weights[scored].sum()),
    }


def unit_scores(units: pd.DataFrame, contract: dict) -> pd.Series:
    """Построчный score единиц: построчная часть формулы mean(E)/wmean(E, w).

    Для формул без построчной части (macro-F1 и т.п.) — совпадение ответа
    агента с размеченной меткой, если в контракте ровно по одному такому входу,
    иначе NaN. Построчный score нужен только дрифт-тестам и примерам в отчёте.
    """
    parsed = formula.parse(contract["formula"])
    unit = parsed.unit_expression()
    if unit is None:
        agent, judged = agent_inputs(contract), judged_inputs(contract)
        if len(agent) == 1 and len(judged) == 1:
            unit = formula.parse(f"{agent[0]['name']} == {judged[0]['name']}")
        else:
            return pd.Series(float("nan"), index=units.index, dtype="float64")
    columns = formula_columns(units, contract)
    missing = [name for name in unit.inputs if name not in columns]
    if missing:
        raise MonitoringContractError(f"Нет входов формулы в единицах оценки: {missing}")
    try:
        return unit.evaluate_rows(columns).astype("float64")
    except formula.FormulaError as exc:
        raise MonitoringContractError(f"Формула КМ: {exc}") from exc


def aggregate_main_metric(frame: pd.DataFrame, payload: dict) -> dict[str, object]:
    """КМ по UMR той же формулой, что baseline на корзине."""
    contract = validate_monitoring_metric(payload)
    result = evaluate_formula(unitize(frame, contract), contract)
    result["name"] = contract["metric_name"]
    return result


def broadcast_scores(frame: pd.DataFrame, units: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    """Разложить score единиц на строки исходного UMR (колонка main_metric)."""
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
