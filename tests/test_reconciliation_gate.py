"""Гейт сверки: baseline публикуется только если формула воспроизводит отчёт."""

from __future__ import annotations

import pytest

from conftest import frame_from, inp, make_layout, make_plan
from laim_basket.errors import NotEvaluableError, ReconciliationError
from laim_basket.llm.generate import _reconciliation_gate
from laim_basket.metric.engine import evaluate


def _gate(frame, layout, plan):
    _scored, km = evaluate(frame, layout, plan)
    _reconciliation_gate(plan, km, frame, layout)
    return km


def test_match_passes():
    km = _gate(frame_from({"Итог": [1, 1, 1, 0]}), make_layout({"E": "Итог"}),
               make_plan("mean(итог)", [inp("E", "итог")], reported="0.75"))
    assert km["reconciliation"]["status"] == "match"


def test_no_reported_value_is_not_gated_here():
    _gate(frame_from({"Итог": [1, 0]}), make_layout({"E": "Итог"}), make_plan("mean(итог)", [inp("E", "итог")]))


def test_mismatch_without_alternative_is_blocking():
    frame = frame_from({"Итог": [1, 0, 0, 0]})
    plan = make_plan("mean(итог)", [inp("E", "итог")], reported="0.82", metric_name="F1")
    with pytest.raises(ReconciliationError) as info:
        _gate(frame, make_layout({"E": "Итог"}), plan)
    assert info.value.reason_code == "km_reconciliation_mismatch"
    assert info.value.details["formula"] == "mean(итог)"
    assert info.value.details["recomputed_value"] == pytest.approx(0.25)


def test_mismatch_with_matching_column_asks_for_repair():
    frame = frame_from({"Черновик": [1, 0, 0, 0], "Итог": [1, 1, 1, 0]})
    layout = make_layout({"D": "Черновик", "E": "Итог"})
    plan = make_plan("mean(x)", [inp("D", "x")], reported="0.75")
    with pytest.raises(NotEvaluableError) as info:
        _gate(frame, layout, plan)
    assert not isinstance(info.value, ReconciliationError)
    matches = info.value.details["matching_mean_columns"]
    assert [m["column_id"] for m in matches] == ["E"] and matches[0]["formula"] == "mean(x)"
