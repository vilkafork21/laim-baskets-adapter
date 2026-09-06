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
        last_data_row=1 + 10,
        roles={},
        grouping={"kind": "none", "column": None},
        dialogue_blob=None,
        weight=(
            {"column_id": weight_column_id, "source": column_names[weight_column_id]}
            if weight_column_id
            else None
        ),
        evidence={},
        column_names=dict(column_names),
        formula_rows={},
        row_ledger=(),
        region=None,
    )


def make_plan(
    method: str,
    sources: list[dict],
    *,
    reducer: str = "mean",
    missing_policy: str = "fail",
    majority_denominator: str | None = None,
    assessment_mode: str = "qa",
    scale: str = "ratio",
    precision: int = 4,
    reported: str | None = None,
    reported_raw: str | None = None,
    threshold: str | None = None,
    comparator: str | None = None,
    metric_name: str = "Accuracy",
    formula: str | None = None,
) -> MeasurementPlan:
    sources = [
        {**source, "name": source.get("name") or f"source_{index}"}
        for index, source in enumerate(sources, start=1)
    ]
    return MeasurementPlan(
        formula=formula,
        basket_id="CI00000001",
        metric_name=metric_name,
        document_roles={
            "instruction": "doc-1",
            "development_report": "doc-2",
            "validation_report": "doc-3",
        },
        assessment_mode=assessment_mode,
        method=method,
        sources=tuple(sources),
        missing_policy=missing_policy,
        majority_denominator=majority_denominator,
        reducer=reducer,
        threshold=Decimal(threshold) if threshold is not None else None,
        comparator=comparator,
        scale=scale,
        precision=precision,
        reported_value=Decimal(reported) if reported is not None else None,
        reported_raw=reported_raw if reported_raw is not None else reported,
        reported_span_id="doc-3:p0001" if reported is not None else None,
        evidence={
            "metric": ("doc-3:p0001",),
            "score": ("doc-1:p0001",),
            "assessment_mode": ("doc-1:p0001",),
            "reducer": ("doc-3:p0002",),
            "missing_policy": (),
            "denominator": (),
            "release": (),
            "reported_value": ("doc-3:p0001",) if reported is not None else (),
        },
    )


def source(column_id: str, role: str, normalization="numeric", polarity: str = "direct", name: str | None = None) -> dict:
    result = {
        "column_id": column_id,
        "role": role,
        "normalization": normalization,
        "polarity": polarity,
    }
    if name:
        result["name"] = name
    return result


def frame_from(columns: dict[str, list], weights: list | None = None) -> pd.DataFrame:
    """Канонический frame корзины: сырые колонки + обязательный input_query_count."""
    size = len(next(iter(columns.values())))
    data = {
        "query_id": [f"q{i}" for i in range(size)],
        "input_query": [f"вопрос {i}" for i in range(size)],
        "output_answer": [f"ответ {i}" for i in range(size)],
        "input_query_count": weights if weights is not None else [1] * size,
    }
    data.update(columns)
    return pd.DataFrame(data)


@pytest.fixture
def layout_factory():
    return make_layout


@pytest.fixture
def plan_factory():
    return make_plan
