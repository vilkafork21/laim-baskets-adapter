"""Регрессии корректности; без реальных клиентских данных и сетевых вызовов."""

from copy import deepcopy
from decimal import Decimal

import pytest
from conftest import layout_answer, metric_answer, source
from helpers import make_workbook

from laim_basket.errors import LayoutError, NotEvaluableError
from laim_basket.llm.tasks import _column_inventory
from laim_basket.metric.baseline import select_baseline
from laim_basket.metric.engine import evaluate
from laim_basket.metric.resolve import resolve_measurement_plan
from laim_basket.reading.xlsx_reader import read_workbook
from laim_basket.resolve import resolve_layout
from laim_basket.transform.canon import build_canon
from laim_basket.transform.grouping import apply_grouping


def materialize(tmp_path, rows, merges=(), proposal=None, sheet="Лист1"):
    path = tmp_path / "input.xlsx"
    make_workbook(path, {sheet: {"rows": rows, "merges": list(merges)}})
    sheets = read_workbook(path)
    layout = resolve_layout(
        proposal or layout_answer(sheet_name=sheet), sheets, "CI09000001", "", frozenset()
    )
    raw = sheets[sheet]
    frame, _ = build_canon(
        apply_grouping(raw, layout.region, layout.transform_config()),
        layout.region,
        layout.transform_config(),
    )
    return layout, frame, raw


def test_decorative_merge_does_not_change_units(tmp_path):
    proposal = layout_answer()
    before = deepcopy(proposal)
    layout, frame, sheet = materialize(
        tmp_path,
        [
            ["q", "a", "m", "category"],
            ["q1", "a1", 1, "one"],
            ["q2", "a2", 1, None],
            ["q3", "a3", 0, "two"],
        ],
        ["D2:D3"],
        proposal,
    )
    plan = resolve_measurement_plan(metric_answer(), layout, frame, sheet)
    _, km = evaluate(frame, layout, plan)
    assert layout.grouping["kind"] == "none"
    assert plan.assessment_mode == "qa"
    assert km["coverage"]["total_units"] == 3
    assert km["recomputed_value"] == pytest.approx(2 / 3)
    assert proposal == before


def test_decorative_merge_does_not_join_separate_dialogues(tmp_path):
    proposal = layout_answer(grouping={"kind": "merged_rows", "column": "A"})
    proposal["roles"].update(session_id="A", input_query="B", output_answer="C")
    layout, frame, sheet = materialize(
        tmp_path,
        [
            ["s", "q", "a", "score", "decoration"],
            ["d1", "q1", "a1", 1, "x"],
            [None, "q2", "a2", None, None],
            ["d2", "q3", "a3", 0, None],
            [None, "q4", "a4", None, None],
        ],
        ["A2:A3", "A4:A5", "D2:D3", "D4:D5", "E2:E5"],
        proposal,
    )
    plan = resolve_measurement_plan(
        metric_answer(sources=[source("D", "final_score")]), layout, frame, sheet
    )
    _, km = evaluate(frame, layout, plan)
    assert km["coverage"]["total_units"] == 2
    assert km["recomputed_value"] == 0.5


def test_shared_score_cannot_label_two_dialogues(tmp_path):
    proposal = layout_answer(grouping={"kind": "merged_rows", "column": "A"})
    proposal["roles"].update(session_id="A", input_query="B", output_answer="C")
    with pytest.raises((LayoutError, NotEvaluableError), match="нескольк|групп"):
        layout, frame, sheet = materialize(
            tmp_path,
            [
                ["s", "q", "a", "score"],
                ["d1", "q1", "a1", 1],
                [None, "q2", "a2", None],
                ["d2", "q3", "a3", None],
                [None, "q4", "a4", None],
            ],
            ["A2:A3", "A4:A5", "D2:D5"],
            proposal,
        )
        resolve_measurement_plan(
            metric_answer(sources=[source("D", "final_score")]), layout, frame, sheet
        )


@pytest.mark.parametrize(
    "proposal",
    [
        metric_answer(method="mean_criteria", sources=[source("C", "criterion")]),
        metric_answer(sources=[source("C", "final_score"), source("D", "criterion")]),
    ],
)
def test_score_method_is_not_silently_rewritten(tmp_path, proposal):
    layout, frame, sheet = materialize(tmp_path, [["q", "a", "m", "m2"], ["q", "a", 1, 0]])
    with pytest.raises(NotEvaluableError):
        resolve_measurement_plan(proposal, layout, frame, sheet)


def test_mixed_percent_tokens_are_not_divided_twice(tmp_path):
    layout, frame, sheet = materialize(
        tmp_path, [["q", "a", "m"], ["q1", "a1", "90%"], ["q2", "a2", 80]]
    )
    plan = resolve_measurement_plan(metric_answer(scale="percent"), layout, frame, sheet)
    scored, km = evaluate(frame, layout, plan)
    assert scored.main_metric.tolist() == [0.9, 0.8]
    assert km["recomputed_value"] == 85


def test_negative_score_is_not_a_valid_ratio(tmp_path):
    layout, frame, sheet = materialize(tmp_path, [["q", "a", "m"], ["q", "a", -0.1]])
    plan = resolve_measurement_plan(metric_answer(), layout, frame, sheet)
    with pytest.raises(NotEvaluableError):
        evaluate(frame, layout, plan)


def test_rare_score_labels_are_visible_to_llm(tmp_path):
    rows = (
        [["q", "a", "m"]] + [[f"q{i}", "a", "yes"] for i in range(9)] + [["q10", "a", "partially"]]
    )
    layout, frame, sheet = materialize(tmp_path, rows)
    inventory = _column_inventory(sheet, frame, layout)
    score = next(x for x in inventory if x["column_id"] == "C")
    assert score["unique_values"] == {"yes": 9, "partially": 1}
    assert score["unique_values_complete"] is True


def test_reference_sheet_cannot_become_queries(tmp_path):
    with pytest.raises(LayoutError, match="справоч|структур"):
        materialize(
            tmp_path,
            [["Параметр", "Значение"], ["Package ID", "CI01"], ["Metric", "Accuracy"]],
            sheet="Описание",
        )


def test_real_queries_on_sheet_named_description_are_valid(tmp_path):
    layout, frame, _ = materialize(
        tmp_path, [["q", "a", "m"], ["Вопрос?", "Ответ", 1]], sheet="Описание"
    )
    assert len(frame) == 1 and layout.sheet_name == "Описание"


def test_two_values_in_one_prose_paragraph_are_not_silently_first():
    candidates = [
        dict(
            metric_name="Accuracy",
            raw=n,
            paragraph="p001",
            kind="value",
            slice_label=None,
            is_key_metric=True,
        )
        for n in ["0.93", "0.71"]
    ]
    result = select_baseline(
        candidates,
        ("Accuracy первой версии 0.93, второй версии 0.71",),
        selected_sheet="Данные",
        metric_name="Accuracy",
        scale="ratio",
    )
    assert result.state == "ambiguous"


def test_rounding_does_not_mask_a_different_metric():
    candidates = [
        dict(
            metric_name=name,
            raw="0.93",
            paragraph=f"p{i:03}",
            kind="value",
            slice_label=None,
            is_key_metric=True,
        )
        for i, name in enumerate(["Accuracy", "Recall"], 1)
    ]
    result = select_baseline(
        candidates, ("Accuracy 0.93", "Recall 0.93"), selected_sheet="Данные", scale="ratio"
    )
    assert result.state == "ambiguous"


def test_grounded_percent_cannot_use_suffix_of_thousands():
    candidates = [
        dict(
            metric_name="Accuracy",
            raw="234",
            paragraph="p001",
            kind="value",
            slice_label=None,
            is_key_metric=True,
        )
    ]
    result = select_baseline(candidates, ("Accuracy 1 234",), selected_sheet="Данные", scale="raw")
    assert result.state == "not_declared"


def test_baseline_batches_keep_tail_and_original_paragraph_ids(monkeypatch):
    from laim_basket import defaults
    from laim_basket.llm.prompts import baseline_batches

    monkeypatch.setattr(defaults, "DOCUMENT_CHAR_CAP", 80)
    report = {
        "port": "validation_report",
        "name": "v.docx",
        "paragraphs": tuple(["Введение " + ("x" * 30)] * 7 + ["Accuracy | 0.93"]),
    }
    batches = list(baseline_batches(report, ["Данные"], "Accuracy"))
    assert len(batches) > 1
    assert "p008: Accuracy | 0.93" in batches[-1][-1]["content"]
    for i in range(1, 9):
        assert any(f"p{i:03d}:" in b[-1]["content"] for b in batches)


def test_evaluation_reconciliation_does_not_rescore(monkeypatch, tmp_path):
    from dataclasses import replace

    from laim_basket.metric import engine

    layout, frame, sheet = materialize(tmp_path, [["q", "a", "m"], ["q", "a", 1]])
    plan = resolve_measurement_plan(metric_answer(), layout, frame, sheet)
    _, km = evaluate(frame, layout, plan)
    monkeypatch.setattr(
        engine, "_sources", lambda *args: (_ for _ in ()).throw(AssertionError("re-scored"))
    )
    updated = engine.reconcile(km, replace(plan, reported_value=Decimal("1.00")))
    assert updated["main_metric"]["value"] == 1.0
    assert updated["reconciliation"]["status"] == "match"


def test_explicit_dialogue_unit_does_not_double_weight_repeated_scores(tmp_path):
    proposal = layout_answer(grouping={"kind": "column", "column": "A"}, weight_column="E")
    proposal["roles"].update(session_id="A", input_query="B", output_answer="C")
    layout, frame, sheet = materialize(
        tmp_path,
        [
            ["session", "q", "a", "score", "frequency"],
            ["d1", "q1", "a1", 1, 2],
            ["d1", "q2", "a2", 1, 2],
            ["d2", "q3", "a3", 0, 3],
            ["d2", "q4", "a4", 0, 3],
            ["d2", "q5", "a5", 0, 3],
        ],
        proposal=proposal,
    )
    plan = resolve_measurement_plan(
        metric_answer(
            assessment_mode="dialogue",
            sources=[source("D", "final_score")],
            quotes={"evaluation_unit": ["Единица оценки — весь диалог."]},
            reducer="frequency_weighted_mean",
        ),
        layout,
        frame,
        sheet,
    )
    _, km = evaluate(frame, layout, plan)
    assert km["coverage"]["total_units"] == 2
    assert km["coverage"]["weight_sum"] == 5
    assert km["recomputed_value"] == 0.4


def test_explicit_dialogue_unit_requires_group_boundaries(tmp_path):
    layout, frame, sheet = materialize(tmp_path, [["q", "a", "m"], ["q", "a", 1]])
    with pytest.raises(NotEvaluableError, match="групп"):
        resolve_measurement_plan(metric_answer(assessment_mode="dialogue"), layout, frame, sheet)


def test_unmapped_text_score_is_not_silently_excluded(tmp_path):
    layout, frame, sheet = materialize(
        tmp_path, [["q", "a", "m"], ["q1", "a1", 1], ["q2", "a2", "частично"]]
    )
    plan = resolve_measurement_plan(metric_answer(), layout, frame, sheet)
    with pytest.raises(NotEvaluableError, match="numeric|числов"):
        evaluate(frame, layout, plan)


def test_mismatch_and_partial_coverage_are_visible_in_warnings(tmp_path):
    from conftest import baseline_answer
    from helpers import FakeClient, make_package

    from laim_basket.pipeline import run_package

    package = make_package(
        tmp_path, {"Лист1": {"rows": [["q", "a", "m"], ["q1", "a1", 1], ["q2", "a2", None]]}}
    )
    result = run_package(
        package,
        tmp_path / "out",
        client=FakeClient([layout_answer(), metric_answer(), baseline_answer()]),
    )
    codes = {w["code"] for w in result.report["warnings"]}
    assert {"reconciliation_mismatch", "score_coverage_partial"} <= codes
    assert result.status == "computed"  # операционный контракт v2 остаётся прежним


def test_declared_non_numeric_missing_token_uses_missing_policy(tmp_path):
    layout, frame, sheet = materialize(
        tmp_path, [["q", "a", "m"], ["q1", "a1", 1], ["q2", "a2", "Рекомендации"]]
    )
    plan = resolve_measurement_plan(
        metric_answer(missing_values={"C": ["Рекомендации"]}), layout, frame, sheet
    )
    scored, km = evaluate(frame, layout, plan)
    assert scored.main_metric.iloc[0] == 1
    assert km["coverage"]["scored_units"] == 1 and km["coverage"]["excluded_units"] == 1
    assert km["recomputed_value"] == 1


def test_numeric_scores_cannot_be_removed_via_missing_tokens(tmp_path):
    layout, frame, sheet = materialize(
        tmp_path, [["q", "a", "m"], ["q1", "a1", 1], ["q2", "a2", 0]]
    )
    with pytest.raises(NotEvaluableError, match="числов"):
        resolve_measurement_plan(metric_answer(missing_values={"C": ["0"]}), layout, frame, sheet)


def test_missing_tokens_do_not_change_external_v2_contract(tmp_path):
    import pandas as pd
    from conftest import baseline_answer
    from helpers import FakeClient, make_package

    from laim_basket.contract import monitoring_metric
    from laim_basket.pipeline import run_package

    package = make_package(
        tmp_path, {"Лист1": {"rows": [["q", "a", "m"], ["q1", "a1", 1], ["q2", "a2", "N/A"]]}}
    )
    result = run_package(
        package,
        tmp_path / "out",
        client=FakeClient(
            [layout_answer(), metric_answer(missing_values={"C": ["N/A"]}), baseline_answer()]
        ),
    )
    contract = monitoring_metric(result)
    assert contract["scoring"]["sources"][0]["normalization"] == "numeric"
    assert set(contract["scoring"]) == {
        "method",
        "sources",
        "missing_policy",
        "majority_denominator",
    }
    assert pd.isna(result.umr.frame["m_metric"].iloc[1])


def test_explicit_grouping_is_not_removed_when_its_role_is_invalid(tmp_path):
    proposal = layout_answer(grouping={"kind": "column", "column": "A"})
    with pytest.raises(LayoutError, match="группировки"):
        materialize(
            tmp_path,
            [["q", "a", "m"], ["same query", "a1", 1], ["same query", "a2", 0]],
            proposal=proposal,
        )
