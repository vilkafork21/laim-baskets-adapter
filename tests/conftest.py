"""Общие фикстуры: план и layout строятся напрямую, без LLM и XLSX."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laim_basket.models import MeasurementPlan, ResolvedLayout  # noqa: E402


def make_layout(column_names: dict[str, str], weight_column_id: str | None = None) -> ResolvedLayout:
    """ResolvedLayout без физического региона: движку нужна только карта колонок."""
    return ResolvedLayout(
        basket_id="CI00000001",
        sheet_name="Лист1",
        ignored_sheets=(),
        header_rows=(1,),
        first_data_row=2,
        last_data_row=11,
        roles={},
        grouping={"kind": "none", "column": None},
        dialogue_blob=None,
        weight=(
            {"column_id": weight_column_id, "source": column_names[weight_column_id]}
            if weight_column_id else None
        ),
        evidence={},
        column_names=dict(column_names),
        formula_rows={},
        row_ledger=(),
        region=None,
    )


def inp(column_id: str, name: str, judged: bool = True) -> dict:
    return {"column_id": column_id, "name": name, "judged": judged}


def make_plan(
    formula: str,
    inputs: list[dict],
    *,
    assessment_mode: str = "qa",
    scale: str = "ratio",
    precision: int = 4,
    reported: str | None = None,
    reported_raw: str | None = None,
    threshold: str | None = None,
    comparator: str | None = None,
    metric_name: str = "Accuracy",
) -> MeasurementPlan:
    return MeasurementPlan(
        basket_id="CI00000001",
        metric_name=metric_name,
        document_roles={"instruction": "doc-1", "development_report": "doc-2", "validation_report": "doc-3"},
        assessment_mode=assessment_mode,
        formula=formula,
        inputs=tuple(inputs),
        threshold=Decimal(threshold) if threshold is not None else None,
        comparator=comparator,
        scale=scale,
        precision=precision,
        reported_value=Decimal(reported) if reported is not None else None,
        reported_raw=reported_raw if reported_raw is not None else reported,
        reported_span_id="doc-3:p0001" if reported is not None else None,
        evidence={
            "metric": ("doc-3:p0001",),
            "formula": ("doc-1:p0001",),
            "assessment_mode": ("doc-1:p0001",),
            "release": (),
            "reported_value": ("doc-3:p0001",) if reported is not None else (),
        },
    )


def frame_from(columns: dict[str, list], weights: list | None = None) -> pd.DataFrame:
    """Канонический frame корзины: сырые колонки + обязательные канонические."""
    size = len(next(iter(columns.values())))
    data = {
        "query_id": [f"q{i}" for i in range(size)],
        "input_query": [f"вопрос {i}" for i in range(size)],
        "output_answer": [f"ответ {i}" for i in range(size)],
        "input_query_count": weights if weights is not None else [1] * size,
    }
    data.update(columns)
    return pd.DataFrame(data)
