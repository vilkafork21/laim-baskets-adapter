"""Движок КМ: домен оценок единиц против объявленной шкалы (аудит LAIM-0189)."""
from __future__ import annotations

import pytest

from conftest import layout_answer, metric_answer, source, evaluation_answer
from helpers import make_workbook
from laim_basket.errors import NotEvaluableError
from laim_basket.metric.engine import evaluate
from laim_basket.metric.resolve import resolve_measurement_plan
from laim_basket.reading.xlsx_reader import read_workbook
from laim_basket.resolve import resolve_layout
from laim_basket.transform.canon import build_canon
from laim_basket.transform.grouping import apply_grouping


def _prepared(tmp_path, scores):
    path = tmp_path / "b.xlsx"
    make_workbook(path, {"Лист1": {"rows": [
        ["q", "a", "m"], *[[f"в{i}", f"о{i}", score] for i, score in enumerate(scores)]]}})
    sheets = read_workbook(path)
    layout = resolve_layout(layout_answer(), sheets, "CI09000001", "", frozenset())
    sheet = sheets["Лист1"]
    grouped = apply_grouping(sheet, layout.region, layout.transform_config())
    frame, _conversion = build_canon(grouped, layout.region, layout.transform_config())
    return layout, frame, sheet


def _percent_plan(layout, frame, sheet, raw: str, normalization="percent"):
    return resolve_measurement_plan(
        metric_answer(scale="percent", evaluation=evaluation_answer(score_values=[0, 0.7, 0.85, 1]), sources=[source("C", "final_score", normalization)],
                      reported_value={"state": "declared", "value": None, "raw": raw}),
        layout, frame, sheet)


def test_percent_point_scores_are_normalized_to_ratio(tmp_path):
    layout, frame, sheet = _prepared(tmp_path, [70, 100, 85])
    plan = _percent_plan(layout, frame, sheet, "85%")

    scored, km = evaluate(frame, layout, plan)

    assert scored["main_metric"].tolist() == [0.7, 1.0, 0.85]
    assert km["main_metric"]["recomputed_value"] == 85.0
    assert km["reconciliation"]["status"] == "match"
    assert km["percent_domain_columns"] == ["C"]


def test_ratio_scores_under_percent_scale_are_left_alone(tmp_path):
    layout, frame, sheet = _prepared(tmp_path, [1, 0, 1])
    plan = _percent_plan(layout, frame, sheet, "66,7%", normalization="numeric")

    scored, km = evaluate(frame, layout, plan)

    assert scored["main_metric"].tolist() == [1.0, 0.0, 1.0]
    assert km["percent_domain_columns"] == []


def test_scores_above_hundred_are_not_a_percent_domain(tmp_path):
    layout, frame, sheet = _prepared(tmp_path, [70, 250, 85])
    plan = _percent_plan(layout, frame, sheet, "85%")

    with pytest.raises(NotEvaluableError):
        evaluate(frame, layout, plan)


def test_percent_point_scores_under_ratio_scale_are_normalized_too(tmp_path):
    # Шкала ratio, а оценки 70/100: доля не бывает больше единицы — это те же
    # процентные пункты, и модель лишь иначе назвала шкалу.
    layout, frame, sheet = _prepared(tmp_path, [70, 100, 85])
    plan = resolve_measurement_plan(
        metric_answer(scale="ratio", evaluation=evaluation_answer(score_values=[0, 0.7, 0.85, 1]), sources=[source("C", "final_score", "percent")],
                      reported_value={"state": "declared", "value": None, "raw": "0.85"}),
        layout, frame, sheet)

    scored, km = evaluate(frame, layout, plan)

    assert scored["main_metric"].tolist() == [0.7, 1.0, 0.85]
    assert km["reconciliation"]["status"] == "match"
    assert km["percent_domain_columns"] == ["C"]


def test_raw_scale_keeps_scores_above_one(tmp_path):
    layout, frame, sheet = _prepared(tmp_path, [2, 1, 2])
    plan = resolve_measurement_plan(
        metric_answer(scale="raw", evaluation=evaluation_answer(score_values=[0, 1, 2]), sources=[source("C", "final_score")],
                      reported_value={"state": "declared", "value": None, "raw": "1.67"}),
        layout, frame, sheet)

    scored, km = evaluate(frame, layout, plan)

    assert scored["main_metric"].tolist() == [2.0, 1.0, 2.0]
    assert km["percent_domain_columns"] == []


@pytest.mark.parametrize("score,expected", [(0.009, "match"), (0.011, "mismatch"),
                                           (0.9, "mismatch")])
def test_small_percent_reconciliation_uses_ratio_quantum(tmp_path, score, expected):
    layout, frame, sheet = _prepared(tmp_path, [score])
    plan = resolve_measurement_plan(
        metric_answer(evaluation=evaluation_answer(score_values=[0, 0.009, 0.011, 0.9, 1]), reported_value={"state": "declared", "value": None, "raw": "0.9%"}),
        layout, frame, sheet)
    _, km = evaluate(frame, layout, plan)
    assert km["reconciliation"]["status"] == expected
