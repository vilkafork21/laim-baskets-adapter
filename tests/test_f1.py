"""Неаддитивные КМ: порядок операций, пропуски, классы и публикация."""

from copy import deepcopy

import pytest
from conftest import baseline_answer, layout_answer, metric_answer, source
from helpers import FakeClient, make_package, make_workbook

from laim_basket.contract import monitoring_metric
from laim_basket.errors import MeasurementPlanError, NotEvaluableError
from laim_basket.metric.engine import evaluate
from laim_basket.metric.resolve import resolve_measurement_plan
from laim_basket.pipeline import run_package
from laim_basket.publish import publish_umr
from laim_basket.reading.xlsx_reader import read_workbook
from laim_basket.resolve import resolve_layout
from laim_basket.transform.canon import build_canon
from laim_basket.transform.grouping import apply_grouping


def harmonic(**changes):
    return metric_answer(
        method="harmonic_mean_of_means",
        sources=[source("C", "precision_component"), source("D", "recall_component")],
        metric_options={"zero_division": 0},
        **changes,
    )


def classification(average="macro", **changes):
    options = changes.pop("metric_options", {})
    return metric_answer(
        method="classification_f1",
        sources=[source("C", "prediction", "label"), source("D", "target", "label")],
        metric_options={
            "average": average,
            "labels": None,
            "positive_label": "yes" if average == "binary" else None,
            "zero_division": 0,
            **options,
        },
        **changes,
    )


def prepared(tmp_path, pairs, proposal, weights=None):
    path = tmp_path / "basket.xlsx"
    rows = [["q", "a", "precision", "recall", "frequency"]]
    rows += [
        [f"в{i}", f"о{i}", left, right, weights[i] if weights else 1]
        for i, (left, right) in enumerate(pairs)
    ]
    make_workbook(path, {"Лист1": {"rows": rows}})
    sheets = read_workbook(path)
    layout = resolve_layout(
        layout_answer(weight_column="E" if weights else None), sheets, "CI09000001", "", frozenset()
    )
    frame, _ = build_canon(
        apply_grouping(sheets["Лист1"], layout.region, layout.transform_config()),
        layout.region,
        layout.transform_config(),
    )
    plan = resolve_measurement_plan(proposal, layout, frame, sheets["Лист1"])
    return layout, frame, plan


def run(tmp_path, pairs, proposal, weights=None):
    layout, frame, plan = prepared(tmp_path, pairs, proposal, weights)
    scored, km = evaluate(frame, layout, plan)
    return plan, scored, km, publish_umr(scored, layout, plan)


def test_harmonic_aggregates_means_not_row_f1(tmp_path):
    _, scored, km, published = run(tmp_path, [(1, 0), (0, 1)], harmonic())
    assert km["recomputed_value"] == 0.5
    assert "main_metric" not in scored and "main_metric" not in published.frame
    stats = km["aggregation_statistics"]
    assert stats["method"] == "harmonic_mean_of_means"
    assert stats["components"]["precision_component"]["mean"] == "0.5"
    assert published.frame["precision_metric"].tolist() == [1.0, 0.0]


def test_harmonic_applies_value_map_once(tmp_path):
    proposal = harmonic()
    for s in proposal["sources"]:
        s["normalization"] = {"0": 0, "1": 0.5, "2": 1}
    plan, _, km, pub = run(tmp_path, [(2, 1), (1, 2)], proposal)
    assert km["recomputed_value"] == 0.75
    assert pub.frame["precision_metric"].tolist() == [1.0, 0.5]
    assert plan.to_dict()["score"]["metric_options"] == {"zero_division": 0}


@pytest.mark.parametrize(
    "policy,expected,units",
    [
        ("exclude_unit", 2 / 3, 1),
        ("exclude_value", 6 / 7, 2),
        ("zero", 0.6, 2),
    ],
)
def test_harmonic_missing_components_are_explicit(tmp_path, policy, expected, units):
    _, _, km, _ = run(tmp_path, [(1, 0.5), (None, 1)], harmonic(missing_policy=policy))
    assert km["recomputed_value"] == pytest.approx(expected)
    assert km["coverage"]["scored_units"] == units


def test_harmonic_missing_fail_rejected(tmp_path):
    with pytest.raises(NotEvaluableError):
        run(tmp_path, [(1, None)], harmonic(missing_policy="fail"))


def test_harmonic_entirely_absent_component_is_not_zero(tmp_path):
    with pytest.raises(NotEvaluableError, match="компонент"):
        run(tmp_path, [(1, None), (0.5, None)], harmonic(missing_policy="exclude_value"))


def test_harmonic_weights_apply_before_harmonic(tmp_path):
    _, _, km, _ = run(
        tmp_path, [(1, 0), (0, 1)], harmonic(reducer="frequency_weighted_mean"), [3, 1]
    )
    assert km["recomputed_value"] == 0.375
    assert km["coverage"]["weight_sum"] == 4


@pytest.mark.parametrize(
    "scale,pairs,expected",
    [
        ("percent", [(100, 50), (50, 100)], 75),
        ("ratio", [("100%", "50%"), ("50%", "100%")], 0.75),
    ],
)
def test_harmonic_percent_domain(tmp_path, scale, pairs, expected):
    assert run(tmp_path, pairs, harmonic(scale=scale))[2]["recomputed_value"] == expected


@pytest.mark.parametrize("zero", [0, 1, "fail"])
def test_harmonic_zero_denominator(tmp_path, zero):
    proposal = harmonic()
    proposal["metric_options"]["zero_division"] = zero
    if zero == "fail":
        with pytest.raises(NotEvaluableError, match="знаменател"):
            run(tmp_path, [(0, 0)], proposal)
    else:
        assert run(tmp_path, [(0, 0)], proposal)[2]["recomputed_value"] == zero


@pytest.mark.parametrize(
    "average,expected",
    [("binary", 2 / 3), ("micro", 0.75), ("macro", 11 / 15), ("weighted", 11 / 15)],
)
def test_classification_f1_variants(tmp_path, average, expected):
    # prediction, target: TP_yes=1, FP_yes=0, FN_yes=1; TP_no=2, FP_no=1, FN_no=0.
    pairs = [("yes", "yes"), ("no", "yes"), ("no", "no"), ("no", "no")]
    _, scored, km, _ = run(tmp_path, pairs, classification(average))
    assert km["recomputed_value"] == pytest.approx(expected)
    assert "main_metric" not in scored
    by_class = {c["label"]: c for c in km["aggregation_statistics"]["per_class"]}
    assert by_class["yes"]["tp"] == "1" and by_class["yes"]["fn"] == "1"


def test_classification_weights_differ_from_class_support_weighting(tmp_path):
    pairs = [("yes", "yes"), ("no", "yes"), ("no", "no")]
    _, _, km, _ = run(
        tmp_path, pairs, classification("binary", reducer="frequency_weighted_mean"), [2, 1, 7]
    )
    assert km["recomputed_value"] == 0.8
    assert km["coverage"]["weight_sum"] == 10


def test_classification_selected_labels_include_outside_errors(tmp_path):
    pairs = [("a", "a"), ("b", "a"), ("a", "b"), ("b", "b")]
    _, _, km, _ = run(tmp_path, pairs, classification("micro", metric_options={"labels": ["a"]}))
    assert km["recomputed_value"] == 0.5
    assert km["aggregation_statistics"]["labels"] == ["a"]


@pytest.mark.parametrize("zero,expected", [(0, 0.5), (1, 1.0)])
def test_classification_absent_class_and_zero_division(tmp_path, zero, expected):
    options = {"labels": ["a", "absent"], "zero_division": zero}
    assert (
        run(tmp_path, [("a", "a")], classification(metric_options=options))[2]["recomputed_value"]
        == expected
    )


def test_classification_errors_are_not_zero_division(tmp_path):
    _, _, km, _ = run(
        tmp_path, [("a", "b"), ("b", "a")], classification(metric_options={"zero_division": 1})
    )
    assert km["recomputed_value"] == 0


def test_classification_normalizes_and_publishes_missing_labels(tmp_path):
    proposal = classification("binary", missing_values={"C": ["unavailable"]})
    _, _, km, pub = run(tmp_path, [(" YES ", "yes"), ("unavailable", "no")], proposal)
    assert km["recomputed_value"] == 1
    assert km["coverage"]["excluded_units"] == 1
    assert pub.frame["precision_output_answer"].tolist() == ["yes", None]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("metric_options"),
        lambda p: p["metric_options"].update(average="samples"),
        lambda p: p["metric_options"].update(labels=[]),
        lambda p: p["metric_options"].update(labels=["A", " a "]),
        lambda p: p.update(missing_policy="zero"),
        lambda p: p.update(missing_policy="exclude_value"),
        lambda p: p.update(scale="raw"),
        lambda p: p["metric_options"].update(positive_label="yes"),
    ],
)
def test_classification_rejects_unspecified_or_inconsistent_definition(tmp_path, mutate):
    proposal = classification()
    mutate(proposal)
    with pytest.raises((MeasurementPlanError, NotEvaluableError)):
        run(tmp_path, [("yes", "yes")], proposal)


def test_binary_cannot_silently_drop_extra_classes(tmp_path):
    with pytest.raises(NotEvaluableError, match="binary"):
        run(tmp_path, [("a", "a"), ("b", "b"), ("yes", "yes")], classification("binary"))


def test_harmonic_disallows_classification_options(tmp_path):
    proposal = harmonic()
    proposal["metric_options"]["average"] = "macro"
    with pytest.raises((MeasurementPlanError, NotEvaluableError)):
        run(tmp_path, [(1, 1)], proposal)


def test_regular_methods_cannot_ignore_metric_options(tmp_path):
    proposal = metric_answer(metric_options={"zero_division": 0})
    with pytest.raises((MeasurementPlanError, NotEvaluableError)):
        run(tmp_path, [(1, 1)], proposal)


def test_harmonic_synthetic_pipeline_preserves_dataset_contract(tmp_path):
    package = make_package(
        tmp_path,
        {
            "Лист1": {
                "rows": [["q", "a", "precision", "recall"], ["в1", "о1", 1, 0], ["в2", "о2", 0, 1]]
            }
        },
    )
    proposals = [layout_answer(), harmonic(), baseline_answer()]
    original = deepcopy(proposals)
    result = run_package(package, tmp_path / "out", client=FakeClient(proposals))
    assert proposals == original
    assert result.status == "computed"
    contract = monitoring_metric(result)
    assert contract["contract_version"] == "laim-monitoring-metric.v2"
    assert contract["score_column"] is None
    assert contract["score_scope"] == "dataset"
    assert contract["aggregation"]["method"] == "harmonic_mean_of_means"
    assert contract["aggregation"]["sample_weighting"] == "uniform"
    assert contract["required_capabilities"] == ["laim.nonadditive-metrics.v1"]
    assert contract["baseline"]["recomputed_value"] == 0.5
    assert result.report["km"]["aggregation_statistics"]["components"]
    assert "main_metric" not in result.umr.frame


def test_classification_synthetic_pipeline_exports_definition(tmp_path):
    package = make_package(
        tmp_path,
        {
            "Лист1": {
                "rows": [
                    ["q", "a", "prediction", "target"],
                    ["в1", "о1", "a", "a"],
                    ["в2", "о2", "b", "a"],
                ]
            }
        },
    )
    result = run_package(
        package,
        tmp_path / "out",
        client=FakeClient(
            [
                layout_answer(),
                classification("micro"),
                baseline_answer(),
            ]
        ),
    )
    contract = monitoring_metric(result)
    assert contract["aggregation"]["method"] == "classification_f1"
    assert contract["scoring"]["metric_options"]["average"] == "micro"
    assert contract["baseline"]["recomputed_value"] == 0.5
    assert result.status == "computed"


def test_class_support_weighted_f1_is_not_macro(tmp_path):
    pairs = [("a", "a")] * 8 + [("a", "b")] + [("b", "b")]
    macro = run(tmp_path, pairs, classification("macro"))[2]["recomputed_value"]
    weighted = run(tmp_path, pairs, classification("weighted"))[2]["recomputed_value"]
    assert macro == pytest.approx((16 / 17 + 2 / 3) / 2)
    assert weighted == pytest.approx((8 * 16 / 17 + 2 * 2 / 3) / 10)
    assert macro != weighted


def test_micro_zero_failure_only_applies_to_global_denominator(tmp_path):
    _, _, km, _ = run(
        tmp_path,
        [("a", "a")],
        classification(
            "micro", metric_options={"labels": ["a", "absent"], "zero_division": "fail"}
        ),
    )
    assert km["recomputed_value"] == 1
    assert km["aggregation_statistics"]["per_class"][1]["f1"] is None


def test_classification_no_complete_pairs_cannot_be_perfect(tmp_path):
    with pytest.raises(NotEvaluableError, match="Ни одной"):
        run(tmp_path, [(None, "a")], classification(metric_options={"zero_division": 1}))


@pytest.mark.parametrize("method", ["harmonic_mean_of_means", "classification_f1"])
def test_nonadditive_dialogue_has_one_weight_per_group(tmp_path, method):
    numeric = method == "harmonic_mean_of_means"
    package = make_package(
        tmp_path,
        {
            "Лист1": {
                "rows": [
                    ["q", "a", "left", "right", "session", "count"],
                    ["в1", "о1", 1 if numeric else "a", 0 if numeric else "a", "d1", 3],
                    ["в2", "о2", 1 if numeric else "a", 0 if numeric else "a", "d1", 3],
                    ["в3", "о3", 0 if numeric else "b", 1 if numeric else "a", "d2", 1],
                ]
            }
        },
    )
    proposal = (harmonic if numeric else classification)(
        assessment_mode="dialogue", reducer="frequency_weighted_mean"
    )
    expected = 0.375 if numeric else 3 / 7
    result = run_package(
        package,
        tmp_path / "out",
        client=FakeClient(
            [
                layout_answer(grouping={"kind": "column", "column": "E"}, weight_column="F"),
                proposal,
                baseline_answer(str(expected)),
            ]
        ),
    )
    assert result.km["recomputed_value"] == pytest.approx(expected)
    assert result.km["coverage"]["total_units"] == 2
    assert result.km["coverage"]["weight_sum"] == 4
    assert len(result.umr.frame) == 2
    assert "main_metric" not in result.umr.frame
    assert result.umr.frame["input_query_count"].tolist() == [3.0, 1.0]


def test_harmonic_degradation_keeps_components_when_baseline_ambiguous(tmp_path):
    package = make_package(
        tmp_path,
        {
            "Лист1": {
                "rows": [
                    ["q", "a", "precision", "recall"],
                    ["в1", "о1", 1, 0.5],
                ]
            }
        },
        validation=("F1 macro 0.76. F1 micro 0.77.",),
    )
    proposal = harmonic()
    proposal["metric_name"] = "F1"
    candidates = baseline_answer("0.76", metric_name="F1 macro") + baseline_answer(
        "0.77", metric_name="F1 micro"
    )
    result = run_package(
        package,
        tmp_path / "out",
        client=FakeClient(
            [
                layout_answer(),
                proposal,
                candidates,
            ]
        ),
    )
    assert result.status == "not_computable"
    assert result.km["reason_code"] == "ambiguous_baseline"
    assert result.report["km"]["aggregation_statistics"]["method"] == "harmonic_mean_of_means"
    assert "precision_metric" in result.umr.frame and "main_metric" not in result.umr.frame
