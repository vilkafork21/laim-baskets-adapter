"""Порт monitoring_metric: формула, входы, единица оценки и доказанный baseline."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import frame_from, inp, make_layout, make_plan
from laim_basket.errors import PackageError
from laim_basket.metric.engine import evaluate
from laim_basket.models import RunResult
from laim_basket.publish import PublishedUmr
from laim_monitoring import validate_monitoring_metric
import main as node


def _result(frame, layout, plan, *, status="computed"):
    scored, km = evaluate(frame, layout, plan)
    published = PublishedUmr(
        variant="flat", sheet_name="Вариант для отд. запросов",
        frame=pd.DataFrame({"main_metric": scored["main_metric"]}),
        published_columns={column_id: f"{name.lower()}_metric" for column_id, name in layout.column_names.items()},
        dropped_non_constant=(),
    )
    return RunResult(status=status, umr=published, km=km, excel_name="umr.xlsx", measurement_plan=plan)


def test_reconciled_plan_yields_valid_contract():
    frame = frame_from({"Итог": [1, 1, 1, 0]})
    layout = make_layout({"E": "Итог"})
    contract = node._monitoring_metric(_result(frame, layout, make_plan("mean(итог)", [inp("E", "итог")], reported="0.75")))
    assert validate_monitoring_metric(contract) == contract
    assert contract["formula"] == "mean(итог)"
    assert contract["inputs"] == [{"name": "итог", "column": "итог_metric", "judged": True}]
    assert contract["baseline"]["value"] == pytest.approx(0.75)
    assert contract["baseline"]["reconciliation"] == "match"
    assert contract["laim_monitoring_version"]


def test_mismatch_never_leaves_node_as_computed():
    frame = frame_from({"Итог": [1, 0, 0, 0]})
    layout = make_layout({"E": "Итог"})
    with pytest.raises(PackageError):
        node._monitoring_metric(_result(frame, layout, make_plan("mean(итог)", [inp("E", "итог")], reported="0.82")))


def test_missing_official_baseline_is_not_computable():
    frame = frame_from({"Итог": [1, 0]})
    contract = node._monitoring_metric(_result(frame, make_layout({"E": "Итог"}), make_plan("mean(итог)", [inp("E", "итог")])))
    assert contract["status"] == "not_computable" and contract["reason_code"] == "official_baseline_missing"
    assert validate_monitoring_metric(contract, require_computed=False)["status"] == "not_computable"


def test_not_evaluable_run_propagates_reason_code():
    result = RunResult(
        status="not_evaluable", umr=PublishedUmr("flat", "x", pd.DataFrame(), {}, ()),
        km={"basket_id": "CI1", "reason": "не воспроизводит", "reason_code": "km_reconciliation_mismatch"},
        excel_name="umr.xlsx", measurement_plan=None,
    )
    contract = node._monitoring_metric(result)
    assert contract["status"] == "not_computable" and contract["reason_code"] == "km_reconciliation_mismatch"
