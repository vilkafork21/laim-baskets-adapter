"""Порт monitoring_metric ноды: computed только с воспроизведённым baseline."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import frame_from, make_layout, make_plan, source
from laim_basket.errors import PackageError
from laim_basket.metric.engine import evaluate
from laim_basket.models import RunResult
from laim_basket.publish import PublishedUmr
import main as node


def _result(frame, layout, plan, *, status="computed"):
    _scored, km = evaluate(frame, layout, plan)
    published = PublishedUmr(
        variant="flat",
        sheet_name="Вариант для отд. запросов",
        frame=pd.DataFrame({"main_metric": _scored["main_metric"]}),
        published_columns={column_id: f"{name}_metric" for column_id, name in layout.column_names.items()},
        dropped_non_constant=(),
    )
    return RunResult(status=status, umr=published, km=km, excel_name="umr.xlsx", measurement_plan=plan)


def test_reconciled_plan_yields_computed_contract():
    frame = frame_from({"Итог": [1, 1, 1, 0]})
    layout = make_layout({"E": "Итог"})
    plan = make_plan("identity", [source("E", "final_score")], reported="0.75")
    contract = node._monitoring_metric(_result(frame, layout, plan))
    assert contract["status"] == "computed"
    assert contract["contract_version"] == "laim-monitoring-metric.v2"  # готовый метод: старые ноды читают
    assert contract["formula"] == "mean(source_1)"
    assert contract["scoring"]["method"] == "identity"
    assert contract["scoring"]["sources"][0]["column_name"] == "Итог_metric"
    assert contract["baseline"]["value"] == pytest.approx(0.75)
    assert contract["baseline"]["recomputed_value"] == pytest.approx(0.75)
    assert contract["baseline"]["reconciliation"] == "match"
    assert contract["primary_validation"]["affects_monitoring"] is False


def test_mismatched_plan_never_leaves_node_as_computed():
    """Гейт не пропускает mismatch; если бы пропустил — нода падает, не публикует."""
    frame = frame_from({"Итог": [1, 0, 0, 0]})
    layout = make_layout({"E": "Итог"})
    plan = make_plan("identity", [source("E", "final_score")], reported="0.82")
    with pytest.raises(PackageError):
        node._monitoring_metric(_result(frame, layout, plan))


def test_missing_official_baseline_is_not_computable():
    frame = frame_from({"Итог": [1, 0]})
    layout = make_layout({"E": "Итог"})
    plan = make_plan("identity", [source("E", "final_score")])
    contract = node._monitoring_metric(_result(frame, layout, plan))
    assert contract["status"] == "not_computable"
    assert contract["reason_code"] == "official_baseline_missing"


def test_not_evaluable_run_propagates_reason_code():
    result = RunResult(
        status="not_evaluable",
        umr=PublishedUmr("flat", "x", pd.DataFrame(), {}, ()),
        km={
            "basket_id": "CI1",
            "reason": "Пересчитанная КМ не воспроизводит значение validation report",
            "reason_code": "km_reconciliation_mismatch",
        },
        excel_name="umr.xlsx",
        measurement_plan=None,
    )
    contract = node._monitoring_metric(result)
    assert contract["status"] == "not_computable"
    assert contract["reason_code"] == "km_reconciliation_mismatch"
