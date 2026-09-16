"""Физика S и отдельное присоединение проверенного baseline K."""

from decimal import Decimal

import pytest
from conftest import baseline_answer, layout_answer, metric_answer, source
from helpers import make_workbook

from laim_basket.errors import MeasurementPlanError, NotEvaluableError
from laim_basket.metric.baseline import attach_baseline, select_baseline
from laim_basket.metric.resolve import reported_quantum, resolve_measurement_plan
from laim_basket.reading.xlsx_reader import read_workbook
from laim_basket.resolve import resolve_layout
from laim_basket.transform.canon import build_canon
from laim_basket.transform.grouping import apply_grouping


@pytest.fixture
def layout_frame(tmp_path):
    path = tmp_path / "b.xlsx"
    make_workbook(
        path, {"Лист1": {"rows": [["q", "a", "m", "m2"], ["в1", "о1", 1, 1], ["в2", "о2", 0, 1]]}}
    )
    sheets = read_workbook(path)
    layout = resolve_layout(layout_answer(), sheets, "CI09000001", "", frozenset())
    sheet = sheets["Лист1"]
    grouped = apply_grouping(sheet, layout.region, layout.transform_config())
    frame, _ = build_canon(grouped, layout.region, layout.transform_config())
    return layout, frame, sheet


def official(raw, text=None, scale="ratio"):
    return select_baseline(
        baseline_answer(raw, "p001"),
        (text or "Accuracy " + raw,),
        selected_sheet="Лист1",
        metric_name="Accuracy",
        scale=scale,
    )


def test_identity_requires_exactly_one_source(layout_frame):
    with pytest.raises(NotEvaluableError):
        resolve_measurement_plan(
            metric_answer(sources=[source("C", "final_score"), source("D", "final_score")]),
            *layout_frame,
        )


def test_unknown_column_rejected(layout_frame):
    with pytest.raises(NotEvaluableError):
        resolve_measurement_plan(
            metric_answer(sources=[source("ZZ", "final_score")]), *layout_frame
        )


def test_weighted_reducer_needs_weight_column(layout_frame):
    with pytest.raises(NotEvaluableError):
        resolve_measurement_plan(metric_answer(reducer="frequency_weighted_mean"), *layout_frame)


def test_threshold_is_not_part_of_score_task(layout_frame):
    with pytest.raises(MeasurementPlanError):
        resolve_measurement_plan(metric_answer(threshold=0.9), *layout_frame)


def test_percent_scale_keeps_declared_domain(layout_frame):
    plan = resolve_measurement_plan(metric_answer(scale="percent"), *layout_frame)
    plan = attach_baseline(plan, official("93", scale="percent"))
    assert plan.scale == "percent" and plan.reported_value == Decimal("93")


def test_bare_value_in_percent_scale_is_not_guessed_as_ratio(layout_frame):
    plan = resolve_measurement_plan(metric_answer(scale="percent"), *layout_frame)
    plan = attach_baseline(plan, official("0.9736", scale="percent"))
    assert plan.reported_value == Decimal("0.9736")


def test_model_cannot_declare_ambiguous_baseline_in_score_task(layout_frame):
    with pytest.raises(MeasurementPlanError):
        resolve_measurement_plan(
            metric_answer(reported_value={"state": "ambiguous"}), *layout_frame
        )


def test_assessment_mode_follows_physical_form(layout_frame):
    plan = resolve_measurement_plan(metric_answer(), *layout_frame)
    assert plan.assessment_mode == "qa" and plan.reported_value is None


def test_reported_citation_found_verbatim():
    assert official("0.987", "Итоговая КМ = 0.987 по результатам").state == "declared"


def test_reported_citation_missing_rejected():
    assert official("0.9", "Ключевая метрика Accuracy равна 0.5").state == "not_declared"


def test_reported_citation_keeps_actual_whitespace():
    assert official("98,7\xa0%", "Итоговая КМ:\n98,7\xa0%").state == "declared"
    assert official("98,7 %", "Итоговая КМ:\n98,7\xa0%").state == "not_declared"


def test_reported_citation_requires_number_boundaries():
    assert official("0.9", "Ключевая метрика равна 0.93").state == "not_declared"
    assert official("0.9", "Ключевая метрика равна 0.9.").state == "declared"


def test_declared_without_raw_rejected():
    result = select_baseline(
        baseline_answer(None, "p001"), ("Accuracy 0.9",), selected_sheet="Лист1"
    )
    assert result.state == "not_declared" and len(result.rejected) == 1


def test_reported_quantum_converts_small_percent_to_ratio(layout_frame):
    plan = resolve_measurement_plan(metric_answer(), *layout_frame)
    plan = attach_baseline(plan, official("0.9%"))
    assert plan.reported_value == Decimal("0.009")
    assert reported_quantum(plan) == Decimal("0.001")
