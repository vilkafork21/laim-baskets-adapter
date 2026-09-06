"""Движок КМ адаптера: формула плана считается общим пакетом на корзине.

Значения посчитаны вручную. Шесть бывших «методов» здесь — обычные формулы.
"""

from __future__ import annotations

import math

import pytest

from conftest import frame_from, inp, make_layout, make_plan
from laim_basket.errors import NotEvaluableError
from laim_basket.metric.engine import evaluate


def test_mean_of_final_score():
    frame = frame_from({"Итог": [1, 0, 1, 1]})
    scored, km = evaluate(frame, make_layout({"E": "Итог"}), make_plan("mean(итог)", [inp("E", "итог")]))
    assert km["status"] == "computed" and km["value_source"] == "recomputed"
    assert km["recomputed_value"] == pytest.approx(0.75)
    assert scored["main_metric"].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert km["coverage"] == {"total_units": 4, "scored_units": 4, "excluded_units": 0, "weight_sum": 4.0}
    assert km["main_metric"]["inputs"] == [{"name": "итог", "column": "Итог", "judged": True}]


def test_weighted_mean_uses_weight_column():
    frame = frame_from({"Итог": [1, 0]}, weights=[3, 1])
    layout = make_layout({"E": "Итог", "F": "freq"}, weight_column_id="F")
    _scored, km = evaluate(frame, layout, make_plan("wmean(итог, weight)", [inp("E", "итог")]))
    assert km["recomputed_value"] == pytest.approx(0.75)
    assert km["coverage"]["weight_sum"] == 4.0


def test_accuracy_compares_labels_case_insensitively():
    frame = frame_from({
        "Класс агента": ["Вклад", "Кредит", "вклад", "Ипотека"],
        "Истинный класс": ["вклад", "Кредит", "Кредит", "Ипотека"],
    })
    layout = make_layout({"C": "Класс агента", "D": "Истинный класс"})
    plan = make_plan("mean(prediction == target)", [inp("C", "prediction", judged=False), inp("D", "target")])
    scored, km = evaluate(frame, layout, plan)
    assert scored["main_metric"].tolist() == [1.0, 1.0, 0.0, 1.0]
    assert km["recomputed_value"] == pytest.approx(0.75)


def test_criteria_formulas():
    frame = frame_from({"Полнота": [1, 1, 0], "Точность": [1, 0, 1]})
    layout = make_layout({"A": "Полнота", "B": "Точность"})
    inputs = [inp("A", "полнота"), inp("B", "точность")]
    _s, km = evaluate(frame, layout, make_plan("mean((полнота + точность) / 2)", inputs))
    assert km["recomputed_value"] == pytest.approx(2 / 3)
    scored, km = evaluate(frame, layout, make_plan("mean(min(полнота, точность))", inputs))
    assert scored["main_metric"].tolist() == [1.0, 0.0, 0.0] and km["recomputed_value"] == pytest.approx(1 / 3)


def test_blank_handling_is_explicit_in_formula():
    frame = frame_from({"A": [1, 1, 0], "B": [1, None, 0]})
    layout = make_layout({"A": "A", "B": "B"})
    inputs = [inp("A", "a"), inp("B", "b")]
    assert evaluate(frame, layout, make_plan("mean((a + b) / 2)", inputs))[1]["recomputed_value"] == pytest.approx(0.5)
    assert evaluate(frame, layout, make_plan("mean(avg(a, b))", inputs))[1]["recomputed_value"] == pytest.approx(2 / 3)
    scored, km = evaluate(frame, layout, make_plan("mean((a + fillna(b, 0)) / 2)", inputs))
    assert scored["main_metric"].tolist() == [1.0, 0.5, 0.0] and km["coverage"]["excluded_units"] == 0


def test_majority_vote():
    frame = frame_from({"A": [1, 1, 1], "B": [1, 0, None], "C": [0, 0, None]})
    layout = make_layout({"A": "A", "B": "B", "C": "C"})
    inputs = [inp("A", "a"), inp("B", "b"), inp("C", "c")]
    _s, km = evaluate(frame, layout, make_plan("mean(majority(a, b, c, declared=True))", inputs))
    assert km["recomputed_value"] == pytest.approx(1 / 3)
    _s, km = evaluate(frame, layout, make_plan("mean(majority(a, b, c))", inputs))
    assert km["recomputed_value"] == pytest.approx(2 / 3)


def test_macro_f1_has_no_rowwise_score_but_publishes_match_trace():
    frame = frame_from({"Класс агента": ["a", "a", "b", "b", "a"], "Истинный класс": ["a", "b", "b", "b", "b"]})
    layout = make_layout({"C": "Класс агента", "D": "Истинный класс"})
    plan = make_plan(
        'f1(prediction, target, "macro")', [inp("C", "prediction", judged=False), inp("D", "target")],
        reported="0.5833", metric_name="Macro F1",
    )
    scored, km = evaluate(frame, layout, plan)
    assert km["recomputed_value"] == pytest.approx(7 / 12)
    assert km["reconciliation"]["status"] == "match"
    assert scored["main_metric"].tolist() == [1.0, 0.0, 1.0, 1.0, 0.0]


def test_threshold_share_formula():
    frame = frame_from({"Оценка": [5, 4, 3, 2]})
    scored, km = evaluate(frame, make_layout({"E": "Оценка"}), make_plan("mean(оценка >= 4)", [inp("E", "оценка")]))
    assert km["recomputed_value"] == pytest.approx(0.5)
    assert scored["main_metric"].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_dialogue_unit_scores_once_per_group_and_requires_constant_input():
    frame = frame_from({"Итог": [1, 1, 0, 0]})
    frame["reference_group_id"] = ["s1", "s1", "s2", "s2"]
    frame["turn_index"] = [1, 2, 1, 2]
    layout = make_layout({"E": "Итог"})
    plan = make_plan("mean(итог)", [inp("E", "итог")], assessment_mode="dialogue")
    scored, km = evaluate(frame, layout, plan)
    assert km["coverage"]["total_units"] == 2 and km["recomputed_value"] == pytest.approx(0.5)
    assert scored["main_metric"].tolist() == [1.0, 1.0, 0.0, 0.0]
    frame.loc[1, "Итог"] = 0
    with pytest.raises(NotEvaluableError):
        evaluate(frame, layout, plan)


def test_unknown_formula_input_is_not_evaluable():
    frame = frame_from({"Итог": [1, 0]})
    with pytest.raises(NotEvaluableError, match="входов формулы"):
        evaluate(frame, make_layout({"E": "Итог"}), make_plan("mean(score)", [inp("E", "итог")]))


def test_reported_value_match_publishes_reported():
    frame = frame_from({"Итог": [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]})
    plan = make_plan("mean(итог)", [inp("E", "итог")], reported="0.90")
    _scored, km = evaluate(frame, make_layout({"E": "Итог"}), plan)
    assert km["reconciliation"]["status"] == "match" and km["value_source"] == "validation_report"
    assert km["main_metric"]["value"] == pytest.approx(0.90)


def test_reported_value_mismatch_is_recorded():
    frame = frame_from({"Итог": [1, 0, 0, 0]})
    _scored, km = evaluate(frame, make_layout({"E": "Итог"}), make_plan("mean(итог)", [inp("E", "итог")], reported="0.82"))
    assert km["reconciliation"]["status"] == "mismatch"
    assert km["reconciliation"]["difference"] == pytest.approx(0.82 - 0.25)


def test_percent_scale_reconciles_ratio_formula_with_percent_report():
    frame = frame_from({"Итог": [1, 1, 1, 0]})
    plan = make_plan("mean(итог)", [inp("E", "итог")], scale="percent", reported="75", reported_raw="75%")
    _scored, km = evaluate(frame, make_layout({"E": "Итог"}), plan)
    assert km["recomputed_value"] == pytest.approx(75.0) and km["reconciliation"]["status"] == "match"


def test_threshold_verdict_is_informational():
    frame = frame_from({"Итог": [1, 1, 0, 0]})
    plan = make_plan("mean(итог)", [inp("E", "итог")], threshold="0.8", comparator=">=")
    _scored, km = evaluate(frame, make_layout({"E": "Итог"}), plan)
    assert km["threshold_verdict"] == "failed" and km["status"] == "computed"
