"""Этап K: модель предлагает упоминания, решение принимает проверяющий код."""

from decimal import Decimal

import pytest

from laim_basket.metric.baseline import select_baseline


def mention(raw="0.93", paragraph="p001", **fields):
    return dict(
        metric_name="Accuracy",
        raw=raw,
        paragraph=paragraph,
        kind="value",
        slice_label=None,
        is_key_metric=True,
        **fields,
    )


def pick(candidates, paragraphs=("Accuracy 0.93",), **kwargs):
    return select_baseline(
        candidates,
        paragraphs,
        selected_sheet=kwargs.pop("sheet", "Данные"),
        metric_name=kwargs.pop("metric_name", "Accuracy"),
        scale=kwargs.pop("scale", "ratio"),
        **kwargs,
    )


def test_general_value_does_not_require_sheet_in_report():
    result = pick([mention()])
    assert result.state == "declared" and result.value == Decimal("0.93")
    assert result.paragraph == "p001" and not result.rejected


@pytest.mark.parametrize(
    "raw,text",
    [
        ("0.9", "Accuracy 0.93"),
        ("9", "Accuracy 0.9"),
        ("0.93", "Accuracy 10.93"),
        ("0.9", "Accuracy 0.9e-3"),
        ("0.93", "Accuracy 0.93%"),
    ],
)
def test_partial_number_citation_is_rejected(raw, text):
    result = pick([mention(raw)], (text,))
    assert result.state == "not_declared" and len(result.rejected) == 1


def test_invalid_candidates_do_not_discard_valid_one():
    result = pick([mention("0.5"), mention(paragraph="p999"), mention()])
    assert result.state == "declared" and len(result.rejected) == 2


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "0.93 abc", "93/100"])
def test_non_numbers_are_rejected(raw):
    result = pick([mention(raw)], ("Accuracy " + raw,))
    assert result.state == "not_declared" and result.rejected


def test_wrong_paragraph_is_not_grounded_elsewhere():
    result = pick([mention(paragraph="p001")], ("Введение", "Accuracy 0.93"))
    assert result.state == "not_declared"


def test_percent_is_an_explicit_domain_even_below_one():
    result = pick([mention("0.9%")], ("Accuracy 0.9%",))
    assert result.scale == "percent" and result.value == Decimal("0.9")


def test_bare_value_uses_score_domain_without_rescaling():
    result = pick([mention("0.97")], ("Accuracy 0.97",), scale="percent")
    assert result.scale == "percent" and result.value == Decimal("0.97")


def test_decimal_comma_and_nbsp_are_parsed():
    result = pick([mention("93,5\xa0%")], ("Accuracy 93,5\xa0%",))
    assert result.value == Decimal("93.5") and result.scale == "percent"


def test_key_metric_precedes_hint():
    candidates = [
        dict(mention(), is_key_metric=False),
        dict(mention("0.85", "p002"), metric_name="F1"),
    ]
    result = pick(candidates, ("Accuracy 0.93", "F1 0.85"))
    assert result.metric_name == "F1" and result.value == Decimal("0.85")


def test_hint_disambiguates_equal_priority_metrics():
    result = pick(
        [mention(), dict(mention("0.85", "p002"), metric_name="F1")], ("Accuracy 0.93", "F1 0.85")
    )
    assert result.value == Decimal("0.93")


def test_slice_matching_is_independent_of_number_of_workbook_sheets():
    candidates = [
        dict(mention("0.93"), kind="slice_value", slice_label="  438  "),
        dict(mention("0.85", "p002"), kind="slice_value", slice_label="554"),
    ]
    result = pick(candidates, ("438 Accuracy 0.93", "554 Accuracy 0.85"), sheet="Агент 438")
    assert result.value == Decimal("0.93")


def test_matching_slice_precedes_general_value():
    result = pick(
        [mention(), dict(mention("0.85", "p002"), kind="slice_value", slice_label="554")],
        ("Accuracy 0.93", "554 Accuracy 0.85"),
        sheet="554",
    )
    assert result.value == Decimal("0.85")


def test_nonmatching_slices_fall_back_to_general_value():
    result = pick(
        [mention(), dict(mention("0.85", "p002"), kind="slice_value", slice_label="554")],
        ("Accuracy 0.93", "554 Accuracy 0.85"),
    )
    assert result.value == Decimal("0.93")


def test_only_nonmatching_slices_are_ambiguous():
    result = pick([dict(mention(), kind="slice_value", slice_label="554")])
    assert result.state == "ambiguous" and result.reason_code == "ambiguous_baseline"


def test_distinct_values_in_different_paragraphs_are_ambiguous():
    result = pick([mention(), mention("0.85", "p002")], ("Accuracy 0.93", "Accuracy 0.85"))
    assert result.state == "ambiguous" and len(result.candidates) == 2


def test_first_number_is_chosen_by_position_not_model_order():
    result = pick(
        [mention("0.85"), mention("0.93"), mention("0.99")], ("Accuracy | 0.93 | 0.85 | 0.99",)
    )
    assert result.value == Decimal("0.93") and len(result.candidates) == 3


def test_rounding_equivalent_values_prefer_most_precise():
    result = pick([mention("0.9"), mention("0.93", "p002")], ("Accuracy 0.9", "Accuracy 0.93"))
    assert result.state == "declared" and result.value == Decimal("0.93")


def test_rounding_not_a_transitive_cluster():
    result = pick(
        [mention("0.90"), mention("0.9", "p002"), mention("0.94", "p003")],
        ("Accuracy 0.90", "Accuracy 0.9", "Accuracy 0.94"),
    )
    assert result.state == "ambiguous"


def test_ratio_and_percent_duplicates_are_equal():
    result = pick([mention("0.93"), mention("93%", "p002")], ("Accuracy 0.93", "Accuracy 93%"))
    assert result.state == "declared"


@pytest.mark.parametrize("kind", ["threshold", "ci_bound", "other"])
def test_non_values_never_become_baseline(kind):
    result = pick([dict(mention(), kind=kind)])
    assert result.state == "not_declared" and result.reason_code == "official_baseline_missing"
    assert len(result.candidates) == 1


def test_threshold_defaults_comparator_and_matches_name():
    result = pick(
        [
            mention(),
            dict(mention("0.8", "p002"), kind="threshold"),
            dict(mention("0.7", "p003"), metric_name="F1", kind="threshold"),
        ],
        ("Accuracy 0.93", "Accuracy порог 0.8", "F1 порог 0.7"),
    )
    assert result.threshold == Decimal("0.8") and result.comparator == ">="


def test_threshold_from_other_slice_is_not_used():
    result = pick(
        [mention(), dict(mention("0.8", "p002"), kind="threshold", slice_label="554")],
        ("Accuracy 0.93", "554 Accuracy порог 0.8"),
    )
    assert result.threshold is None


def test_empty_response_is_explicit_missing():
    result = pick([])
    assert result.state == "not_declared" and result.value is None
    assert all(word in result.reason for word in ("Этап K", "Ожидалось", "Получено", "Действие"))


def test_threshold_conflict_is_visible_and_does_not_invent_verdict():
    result = pick(
        [
            mention(),
            dict(mention("0.8", "p002"), kind="threshold"),
            dict(mention("0.9", "p003"), kind="threshold", comparator="<="),
        ],
        ("Accuracy 0.93", "Accuracy порог 0.8", "Accuracy порог 0.9"),
    )
    assert result.state == "declared" and result.threshold is None
    assert result.warnings[0]["code"] == "threshold_ambiguous"


@pytest.mark.parametrize("raw", ["1e9999", "1e-9999"])
def test_values_not_representable_in_json_number_are_rejected(raw):
    result = pick([mention(raw)], ("Accuracy " + raw,))
    assert result.state == "not_declared" and result.rejected
