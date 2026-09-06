"""Движок КМ: каждая формула реестра считается детерминированно и проверяемо.

Значения ниже посчитаны вручную. Тест фиксирует контракт: если формула метода
или политика пропусков изменится, число изменится и тест это покажет.
"""

from __future__ import annotations

import math

import pytest

from conftest import frame_from, make_layout, make_plan, source
from laim_basket.errors import NotEvaluableError
from laim_basket.metric.engine import evaluate


def _km(frame, layout, plan):
    scored, km = evaluate(frame, layout, plan)
    return scored, km


def test_identity_mean_is_plain_average_of_scores():
    frame = frame_from({"Итог": [1, 0, 1, 1]})
    layout = make_layout({"E": "Итог"})
    plan = make_plan("identity", [source("E", "final_score")])
    scored, km = _km(frame, layout, plan)
    assert km["status"] == "computed"
    assert km["recomputed_value"] == pytest.approx(0.75)
    assert km["value_source"] == "recomputed"
    assert scored["main_metric"].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert km["coverage"] == {
        "total_units": 4, "scored_units": 4, "excluded_units": 0, "weight_sum": 4.0,
    }


def test_identity_frequency_weighted_mean_uses_input_query_count():
    frame = frame_from({"Итог": [1, 0]}, weights=[3, 1])
    layout = make_layout({"E": "Итог", "F": "freq"}, weight_column_id="F")
    plan = make_plan(
        "identity", [source("E", "final_score")], reducer="frequency_weighted_mean",
    )
    _scored, km = _km(frame, layout, plan)
    assert km["recomputed_value"] == pytest.approx(0.75)
    assert km["main_metric"]["weighted"] is True
    assert km["coverage"]["weight_sum"] == 4.0


def test_accuracy_compares_labels_case_insensitively():
    frame = frame_from({
        "Класс агента": ["Вклад", "Кредит", "вклад", "Ипотека"],
        "Истинный класс": ["вклад", "Кредит", "Кредит", "Ипотека"],
    })
    layout = make_layout({"C": "Класс агента", "D": "Истинный класс"})
    plan = make_plan(
        "accuracy",
        [source("C", "prediction", "label"), source("D", "target", "label")],
    )
    scored, km = _km(frame, layout, plan)
    assert scored["main_metric"].tolist() == [1.0, 1.0, 0.0, 1.0]
    assert km["recomputed_value"] == pytest.approx(0.75)


def test_mean_criteria_averages_criteria_per_row():
    frame = frame_from({"Полнота": [1, 0], "Точность": [1, 1], "Ясность": [0, 1]})
    layout = make_layout({"A1": "Полнота", "B1": "Точность", "C1": "Ясность"})
    plan = make_plan(
        "mean_criteria",
        [source("A1", "criterion"), source("B1", "criterion"), source("C1", "criterion")],
    )
    scored, km = _km(frame, layout, plan)
    assert scored["main_metric"].tolist() == pytest.approx([2 / 3, 2 / 3])
    assert km["recomputed_value"] == pytest.approx(2 / 3)


def test_all_criteria_is_conjunction_of_binary_criteria():
    frame = frame_from({"Полнота": [1, 1, 0], "Точность": [1, 0, 1]})
    layout = make_layout({"A1": "Полнота", "B1": "Точность"})
    plan = make_plan(
        "all_criteria", [source("A1", "criterion"), source("B1", "criterion")],
    )
    scored, km = _km(frame, layout, plan)
    assert scored["main_metric"].tolist() == [1.0, 0.0, 0.0]
    assert km["recomputed_value"] == pytest.approx(1 / 3)


def test_majority_declared_denominator_counts_absent_votes_as_negative():
    frame = frame_from({"A": [1, 1, 1], "B": [1, 0, None], "C": [0, 0, None]})
    layout = make_layout({"A": "A", "B": "B", "C": "C"})
    votes = [source("A", "assessor_vote"), source("B", "assessor_vote"), source("C", "assessor_vote")]
    plan = make_plan(
        "majority", votes, missing_policy="exclude_unit", majority_denominator="declared",
    )
    scored, km = _km(frame, layout, plan)
    # строка 3: один голос «за» из трёх заявленных — не большинство
    assert scored["main_metric"].tolist() == [1.0, 0.0, 0.0]
    assert km["recomputed_value"] == pytest.approx(1 / 3)


def test_majority_present_denominator_and_tie_follows_missing_policy():
    frame = frame_from({"A": [1, 1], "B": [0, None]})
    layout = make_layout({"A": "A", "B": "B"})
    votes = [source("A", "assessor_vote"), source("B", "assessor_vote")]
    plan = make_plan(
        "majority", votes, missing_policy="exclude_unit", majority_denominator="present",
    )
    scored, km = _km(frame, layout, plan)
    # строка 1: ничья 1:1 → единица исключена; строка 2: 1 из 1 присутствующего
    assert math.isnan(scored["main_metric"].tolist()[0])
    assert scored["main_metric"].tolist()[1] == 1.0
    assert km["coverage"]["excluded_units"] == 1
    assert km["recomputed_value"] == pytest.approx(1.0)


def test_all_assessors_requires_unanimity():
    frame = frame_from({"A": [1, 1, 0], "B": [1, 0, 0]})
    layout = make_layout({"A": "A", "B": "B"})
    plan = make_plan(
        "all_assessors", [source("A", "assessor_vote"), source("B", "assessor_vote")],
    )
    scored, km = _km(frame, layout, plan)
    assert scored["main_metric"].tolist() == [1.0, 0.0, 0.0]
    assert km["recomputed_value"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "policy, expected_scores, expected_value",
    [
        ("exclude_unit", [1.0, None, 0.0], 0.5),
        ("exclude_value", [1.0, 1.0, 0.0], 2 / 3),
        ("zero", [1.0, 0.5, 0.0], 0.5),
    ],
)
def test_missing_policy_for_mean_criteria(policy, expected_scores, expected_value):
    frame = frame_from({"A": [1, 1, 0], "B": [1, None, 0]})
    layout = make_layout({"A": "A", "B": "B"})
    plan = make_plan(
        "mean_criteria", [source("A", "criterion"), source("B", "criterion")],
        missing_policy=policy,
    )
    scored, km = _km(frame, layout, plan)
    actual = [None if math.isnan(v) else v for v in scored["main_metric"].tolist()]
    assert actual == expected_scores
    assert km["recomputed_value"] == pytest.approx(expected_value)


def test_missing_policy_fail_raises():
    frame = frame_from({"Итог": [1, None]})
    layout = make_layout({"E": "Итог"})
    plan = make_plan("identity", [source("E", "final_score")], missing_policy="fail")
    with pytest.raises(NotEvaluableError):
        evaluate(frame, layout, plan)


def test_inverted_polarity_flips_binary_score():
    frame = frame_from({"Ошибка": [1, 0, 0]})
    layout = make_layout({"E": "Ошибка"})
    plan = make_plan("identity", [source("E", "final_score", "numeric", "inverted")])
    scored, km = _km(frame, layout, plan)
    assert scored["main_metric"].tolist() == [0.0, 1.0, 1.0]
    assert km["recomputed_value"] == pytest.approx(2 / 3)


def test_value_map_normalization_maps_labels_to_scores():
    frame = frame_from({"Оценка": ["Да", "Нет", "да"]})
    layout = make_layout({"E": "Оценка"})
    plan = make_plan("identity", [source("E", "final_score", {"Да": 1, "Нет": 0})])
    scored, km = _km(frame, layout, plan)
    assert scored["main_metric"].tolist() == [1.0, 0.0, 1.0]


def test_dialogue_unit_scores_once_per_group_and_requires_constant_source():
    frame = frame_from({"Итог": [1, 1, 0, 0]})
    frame["reference_group_id"] = ["s1", "s1", "s2", "s2"]
    frame["turn_index"] = [1, 2, 1, 2]
    layout = make_layout({"E": "Итог"})
    plan = make_plan("identity", [source("E", "final_score")], assessment_mode="dialogue")
    scored, km = _km(frame, layout, plan)
    assert km["coverage"]["total_units"] == 2
    assert km["recomputed_value"] == pytest.approx(0.5)
    assert scored["main_metric"].tolist() == [1.0, 1.0, 0.0, 0.0]

    frame.loc[1, "Итог"] = 0
    with pytest.raises(NotEvaluableError):
        evaluate(frame, layout, plan)


def test_reported_value_match_publishes_reported_and_marks_match():
    frame = frame_from({"Итог": [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]})
    layout = make_layout({"E": "Итог"})
    plan = make_plan(
        "identity", [source("E", "final_score")], reported="0.90", reported_raw="0.90",
    )
    _scored, km = _km(frame, layout, plan)
    assert km["reconciliation"]["status"] == "match"
    assert km["value_source"] == "validation_report"
    assert km["main_metric"]["value"] == pytest.approx(0.90)


def test_reported_value_mismatch_is_recorded_by_engine():
    frame = frame_from({"Итог": [1, 0, 0, 0]})
    layout = make_layout({"E": "Итог"})
    plan = make_plan(
        "identity", [source("E", "final_score")], reported="0.82", reported_raw="0.82",
    )
    _scored, km = _km(frame, layout, plan)
    assert km["reconciliation"]["status"] == "mismatch"
    assert km["reconciliation"]["difference"] == pytest.approx(0.82 - 0.25)


def test_percent_scale_reconciles_ratio_frame_with_percent_report():
    frame = frame_from({"Итог": [1, 1, 1, 0]})
    layout = make_layout({"E": "Итог"})
    plan = make_plan(
        "identity", [source("E", "final_score")],
        scale="percent", reported="75", reported_raw="75%",
    )
    _scored, km = _km(frame, layout, plan)
    assert km["recomputed_value"] == pytest.approx(75.0)
    assert km["reconciliation"]["status"] == "match"


def test_threshold_verdict_is_informational():
    frame = frame_from({"Итог": [1, 1, 0, 0]})
    layout = make_layout({"E": "Итог"})
    plan = make_plan(
        "identity", [source("E", "final_score")], threshold="0.8", comparator=">=",
    )
    _scored, km = _km(frame, layout, plan)
    assert km["threshold_verdict"] == "failed"
    assert km["status"] == "computed"


def test_formula_plan_computes_macro_f1_as_in_report():
    frame = frame_from({
        "Класс агента": ["a", "a", "b", "b", "a"],
        "Истинный класс": ["a", "b", "b", "b", "b"],
    })
    layout = make_layout({"C": "Класс агента", "D": "Истинный класс"})
    plan = make_plan(
        "formula",
        [
            source("C", "prediction", "label", name="prediction"),
            source("D", "target", "label", name="target"),
        ],
        formula='f1(prediction, target, "macro")',
        reported="0.5833", reported_raw="0.5833", metric_name="Macro F1",
    )
    scored, km = evaluate(frame, layout, plan)
    # a: P=1/3 R=1 F1=0.5; b: P=1 R=0.5 F1=2/3 → macro 7/12
    assert km["recomputed_value"] == pytest.approx(7 / 12)
    assert km["reconciliation"]["status"] == "match"
    assert km["main_metric"]["formula"] == 'f1(prediction, target, "macro")'
    # построчного score у F1 нет — публикуется совпадение с истинным классом
    assert scored["main_metric"].tolist() == [1.0, 0.0, 1.0, 1.0, 0.0]


def test_formula_plan_threshold_share():
    frame = frame_from({"Оценка": [5, 4, 3, 2]})
    layout = make_layout({"E": "Оценка"})
    plan = make_plan(
        "formula", [source("E", "final_score", name="оценка")], formula="mean(оценка >= 4)",
    )
    scored, km = evaluate(frame, layout, plan)
    assert km["recomputed_value"] == pytest.approx(0.5)
    assert scored["main_metric"].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_preset_methods_are_published_as_formulas():
    frame = frame_from({"A": [1, 1], "B": [1, 0]})
    layout = make_layout({"A": "A", "B": "B"})
    plan = make_plan("mean_criteria", [source("A", "criterion"), source("B", "criterion")],
                     missing_policy="exclude_value")
    _scored, km = evaluate(frame, layout, plan)
    assert km["main_metric"]["formula"] == "mean(avg(source_1, source_2))"
