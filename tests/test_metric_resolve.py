"""Физика плана: контракт источников, формулы, веса, шкалы, режим оценки."""
from __future__ import annotations

from decimal import Decimal


import pytest


from conftest import layout_answer, metric_answer, source
from helpers import make_workbook
from laim_basket.errors import (
    AmbiguousBaselineError,
    MeasurementPlanError,
    NotEvaluableError,
)
from laim_basket.metric.resolve import (
    reported_quantum,
    resolve_measurement_plan,
    verify_reported_citation,
)
from laim_basket.reading.xlsx_reader import read_workbook
from laim_basket.resolve import resolve_layout
from laim_basket.transform.canon import build_canon
from laim_basket.transform.grouping import apply_grouping


@pytest.fixture
def layout_frame(tmp_path):
    path = tmp_path / "b.xlsx"
    make_workbook(path, {"Лист1": {"rows": [
        ["q", "a", "m", "m2"], ["в1", "о1", 1, 1], ["в2", "о2", 0, 1]]}})
    sheets = read_workbook(path)
    layout = resolve_layout(layout_answer(), sheets, "CI09000001", "", frozenset())
    sheet = sheets["Лист1"]
    grouped = apply_grouping(sheet, layout.region, layout.transform_config())
    frame, _conversion = build_canon(grouped, layout.region, layout.transform_config())
    return layout, frame, sheet


def test_identity_requires_exactly_one_source(layout_frame):
    layout, frame, sheet = layout_frame
    proposal = metric_answer(sources=[
        source("C", "final_score"),
        source("D", "final_score")])
    with pytest.raises(NotEvaluableError):
        resolve_measurement_plan(proposal, layout, frame, sheet)


def test_unknown_column_rejected(layout_frame):
    layout, frame, sheet = layout_frame
    proposal = metric_answer(sources=[
        source("ZZ", "final_score")])
    with pytest.raises(NotEvaluableError):
        resolve_measurement_plan(proposal, layout, frame, sheet)


def test_weighted_reducer_needs_weight_column(layout_frame):
    layout, frame, sheet = layout_frame
    with pytest.raises(NotEvaluableError):
        resolve_measurement_plan(
            metric_answer(reducer="frequency_weighted_mean"),
            layout, frame, sheet)


def test_threshold_without_comparator_rejected(layout_frame):
    layout, frame, sheet = layout_frame
    with pytest.raises(MeasurementPlanError):
        resolve_measurement_plan(
            metric_answer(threshold=0.9), layout, frame, sheet)


def test_percent_scale_keeps_declared_domain(layout_frame):
    layout, frame, sheet = layout_frame
    plan = resolve_measurement_plan(
        metric_answer(scale="percent",
                          reported_value={"state": "declared", "value": 93.0,
                                           "raw": "93"}),
        layout, frame, sheet)
    assert plan.scale == "percent" and float(plan.reported_value) == 93.0


def test_bare_share_in_percent_scale_normalized(layout_frame):
    layout, frame, sheet = layout_frame
    plan = resolve_measurement_plan(
        metric_answer(scale="percent",
                          reported_value={"state": "declared", "value": 0.9736,
                                           "raw": "0.9736"}),
        layout, frame, sheet)
    assert float(plan.reported_value) == 97.36


def test_ambiguous_baseline_degrades(layout_frame):
    layout, frame, sheet = layout_frame
    with pytest.raises(AmbiguousBaselineError):
        resolve_measurement_plan(
            metric_answer(reported_value={"state": "ambiguous", "value": None,
                                              "raw": None}),
            layout, frame, sheet)


def test_assessment_mode_follows_physical_form(layout_frame):
    layout, frame, sheet = layout_frame
    plan = resolve_measurement_plan(metric_answer(), layout, frame, sheet)
    assert plan.assessment_mode == "qa"


def test_reported_citation_found_verbatim():
    verify_reported_citation("0.987", "Итоговая КМ = 0.987 по результатам")


def test_reported_citation_missing_rejected():
    with pytest.raises(MeasurementPlanError):
        verify_reported_citation("0.9", "Ключевая метрика Accuracy равна 0.5")


def test_reported_citation_ignores_whitespace_runs():
    verify_reported_citation("98,7 %", "Итоговая КМ:\n98,7\xa0%")


def test_reported_citation_requires_number_boundaries():
    with pytest.raises(MeasurementPlanError):
        verify_reported_citation("0.9", "Ключевая метрика равна 0.93")
    verify_reported_citation("0.9", "Ключевая метрика равна 0.9.")


def test_declared_without_raw_rejected(layout_frame):
    layout, frame, sheet = layout_frame
    with pytest.raises(MeasurementPlanError):
        resolve_measurement_plan(
            metric_answer(reported_value={"state": "declared", "value": 0.9,
                                          "raw": None}),
            layout, frame, sheet)


@pytest.mark.parametrize("raw,scale,value,quantum", [
    ("0.9%", "ratio", "0.009", "0.001"),
    ("1%", "ratio", "0.01", "0.01"),
    ("0%", "ratio", "0", "0.01"),
    ("90%", "ratio", "0.9", "0.01"),
    ("0.9%", "percent", "0.9", "0.1"),
    ("90%", "percent", "90", "1"),
])
def test_explicit_percent_value_and_quantum(layout_frame, raw, scale, value, quantum):
    layout, frame, sheet = layout_frame
    plan = resolve_measurement_plan(
        metric_answer(scale=scale, reported_value={"state": "declared", "value": None,
                                                   "raw": raw}),
        layout, frame, sheet)
    assert plan.reported_value == Decimal(value)
    assert reported_quantum(plan) == Decimal(quantum)
