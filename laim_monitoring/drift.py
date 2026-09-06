"""Кадры для дрифт-тестов: вопрос, построчный target и группа."""

from __future__ import annotations

import json

import pandas as pd

from .contract import validate_monitoring_metric
from .errors import MonitoringContractError
from .units import _unitize, load_monitoring_frame
from .values import blank


def _drift_frame(frame: pd.DataFrame, contract: dict, *, require_target: bool) -> pd.DataFrame:
    units = _unitize(frame, contract)
    if require_target and "main_metric" not in units:
        raise MonitoringContractError("Для drift отсутствует main_metric")
    target = pd.to_numeric(
        units["main_metric"] if "main_metric" in units else pd.Series([None] * len(units)),
        errors="coerce",
    )
    missing_policy = contract.get("scoring", {}).get("missing_policy", "fail")
    if require_target and missing_policy == "fail" and target.isna().any():
        raise MonitoringContractError("Для drift обнаружен пустой main_metric")
    mode = contract["assessment_mode"]
    if mode == "dialogue":
        questions = [
            json.dumps(
                [turn["input_query"] for turn in dialogue],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for dialogue in units["dialogue"]
        ]
    elif mode == "turn_with_history":
        questions = [
            json.dumps(
                [
                    *(turn["input_query"] for turn in context["history"]),
                    context["current_turn"]["input_query"],
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for context in units["assessment_context"]
        ]
    else:
        questions = units["input_query"].astype(str).tolist()
    group_ids = [
        unit_id if blank(group_id) else group_id
        for group_id, unit_id in zip(units["_group_id"], units["_unit_id"])
    ]
    result = pd.DataFrame({
        "question": questions,
        "answer": [""] * len(units),
        "target": target,
        "reference_group_id": group_ids,
    })
    return result.dropna(subset=["target"]).reset_index(drop=True) if require_target else result


def prepare_drift_frames(
    reference_umr: pd.DataFrame,
    monitoring_umr: pd.DataFrame | bytes | bytearray | str | Path,
    payload: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contract = validate_monitoring_metric(payload, require_computed=False)
    if "assessment_mode" not in contract:
        # Контракт без MeasurementPlan (plan-less not_computable) не несёт
        # режим оценки — drift честно отказывается вместо невнятного падения
        # внутри нормализации.
        raise MonitoringContractError(
            "Drift не вычисляется: monitoring_metric не содержит assessment_mode "
            f"(status={contract.get('status')!r}, reason={contract.get('reason')!r})"
        )
    # normalize_umr выполнит _unitize внутри _drift_frame — второй проход не нужен
    monitoring_umr = load_monitoring_frame(monitoring_umr)
    return (
        _drift_frame(reference_umr, contract, require_target=True),
        _drift_frame(monitoring_umr, contract, require_target=False),
    )
