"""Движок КМ: домен оценок единиц против объявленной шкалы (аудит LAIM-0189)."""

from __future__ import annotations

import pytest
from conftest import baseline_answer, layout_answer, metric_answer
from helpers import make_workbook

from laim_basket.errors import NotEvaluableError
from laim_basket.metric.baseline import attach_baseline, select_baseline
from laim_basket.metric.engine import evaluate
from laim_basket.metric.resolve import resolve_measurement_plan
from laim_basket.reading.xlsx_reader import read_workbook
from laim_basket.resolve import resolve_layout
from laim_basket.transform.canon import build_canon
from laim_basket.transform.grouping import apply_grouping


def _prepared(tmp_path, scores):
    path = tmp_path / "b.xlsx"
    make_workbook(
        path,
        {
            "Лист1": {
                "rows": [
                    ["q", "a", "m"],
                    *[[f"в{i}", f"о{i}", score] for i, score in enumerate(scores)],
                ]
            }
        },
    )
    sheets = read_workbook(path)
    layout = resolve_layout(layout_answer(), sheets, "CI09000001", "", frozenset())
    sheet = sheets["Лист1"]
    grouped = apply_grouping(sheet, layout.region, layout.transform_config())
    frame, _conversion = build_canon(grouped, layout.region, layout.transform_config())
    return layout, frame, sheet


def _reported_plan(layout, frame, sheet, raw: str, scale="percent"):
    plan = resolve_measurement_plan(metric_answer(scale=scale), layout, frame, sheet)
    baseline = select_baseline(
        baseline_answer(raw, "p001"),
        ("Accuracy " + raw,),
        selected_sheet=layout.sheet_name,
        metric_name="Accuracy",
        scale=scale,
    )
    return attach_baseline(plan, baseline)


def _percent_plan(layout, frame, sheet, raw: str):
    return _reported_plan(layout, frame, sheet, raw)


def test_percent_point_scores_are_normalized_to_ratio(tmp_path):
    layout, frame, sheet = _prepared(tmp_path, [70, 100, 85])
    plan = _percent_plan(layout, frame, sheet, "85%")

    scored, km = evaluate(frame, layout, plan)

    assert scored["main_metric"].tolist() == [0.7, 1.0, 0.85]
    assert km["main_metric"]["recomputed_value"] == 85.0
    assert km["reconciliation"]["status"] == "match"
    assert km["percent_domain_columns"] == ["C"]


def test_bare_values_under_percent_scale_are_percent_points(tmp_path):
    layout, frame, sheet = _prepared(tmp_path, [1, 0, 1])
    plan = _percent_plan(layout, frame, sheet, "66,7%")

    scored, km = evaluate(frame, layout, plan)

    assert scored["main_metric"].tolist() == [0.01, 0.0, 0.01]
    assert km["percent_domain_columns"] == ["C"]


def test_scores_above_hundred_are_not_a_percent_domain(tmp_path):
    layout, frame, sheet = _prepared(tmp_path, [70, 250, 85])
    plan = _percent_plan(layout, frame, sheet, "85%")

    with pytest.raises(NotEvaluableError):
        evaluate(frame, layout, plan)


def test_percent_points_under_ratio_scale_require_repair(tmp_path):
    # Не угадываем шкалу по максимуму: одна новая строка не меняет прежние оценки.
    layout, frame, sheet = _prepared(tmp_path, [70, 100, 85])
    plan = _reported_plan(layout, frame, sheet, "0.85", "ratio")
    with pytest.raises(NotEvaluableError, match="явной шкале"):
        evaluate(frame, layout, plan)


def test_raw_scale_keeps_scores_above_one(tmp_path):
    layout, frame, sheet = _prepared(tmp_path, [2, 1, 2])
    plan = _reported_plan(layout, frame, sheet, "1.67", "raw")

    scored, km = evaluate(frame, layout, plan)

    assert scored["main_metric"].tolist() == [2.0, 1.0, 2.0]
    assert km["percent_domain_columns"] == []
