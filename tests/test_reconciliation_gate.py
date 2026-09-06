"""Гейт сверки: baseline публикуется только если план воспроизводит отчёт.

Ровно тот путь, который раньше молча публиковал «заявленное значение с пометкой
mismatch» и отправлял вниз несопоставимый baseline.
"""

from __future__ import annotations

import pytest

from conftest import frame_from, make_layout, make_plan, source
from laim_basket.errors import NotEvaluableError, ReconciliationError
from laim_basket.llm.generate import _reconciliation_gate
from laim_basket.metric.engine import evaluate


def test_match_passes_silently():
    frame = frame_from({"Итог": [1, 1, 1, 0]})
    layout = make_layout({"E": "Итог"})
    plan = make_plan("identity", [source("E", "final_score")], reported="0.75")
    _scored, km = evaluate(frame, layout, plan)
    _reconciliation_gate(plan, km, frame, layout)


def test_no_reported_value_is_not_gated_here():
    frame = frame_from({"Итог": [1, 0]})
    layout = make_layout({"E": "Итог"})
    plan = make_plan("identity", [source("E", "final_score")])
    _scored, km = evaluate(frame, layout, plan)
    _reconciliation_gate(plan, km, frame, layout)


def test_mismatch_without_alternative_is_blocking():
    """Метрика отчёта не воспроизводится ни одной колонкой: отказ, не warning."""
    frame = frame_from({"Итог": [1, 0, 0, 0]})
    layout = make_layout({"E": "Итог"})
    plan = make_plan(
        "identity", [source("E", "final_score")],
        reported="0.82", reported_raw="0.82", metric_name="F1",
    )
    _scored, km = evaluate(frame, layout, plan)
    assert km["reconciliation"]["status"] == "mismatch"
    with pytest.raises(ReconciliationError) as info:
        _reconciliation_gate(plan, km, frame, layout)
    assert info.value.reason_code == "km_reconciliation_mismatch"
    assert info.value.details["metric_name"] == "F1"
    assert info.value.details["reported_value"] == "0.82"
    assert info.value.details["recomputed_value"] == pytest.approx(0.25)


def test_mismatch_with_matching_identity_plan_asks_for_repair():
    """Выбран не тот score, но в корзине есть колонка, дающая число отчёта."""
    frame = frame_from({"Черновик": [1, 0, 0, 0], "Итог": [1, 1, 1, 0]})
    layout = make_layout({"D": "Черновик", "E": "Итог"})
    plan = make_plan(
        "identity", [source("D", "final_score")], reported="0.75", reported_raw="0.75",
    )
    _scored, km = evaluate(frame, layout, plan)
    with pytest.raises(NotEvaluableError) as info:
        _reconciliation_gate(plan, km, frame, layout)
    assert not isinstance(info.value, ReconciliationError)
    matches = info.value.details["matching_identity_plans"]
    assert [m["column_id"] for m in matches] == ["E"]
    assert matches[0]["recomputed"] == pytest.approx(0.75)
