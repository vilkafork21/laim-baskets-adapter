"""Контракты портов: monitoring_metric v2 неприкосновенен, km_result = run-report."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from conftest import layout_answer, metric_answer
from helpers import FakeClient, make_package
from laim_basket.errors import LayoutError

MODULE_DIR = Path(__file__).resolve().parents[1]

ROWS = [["q", "a", "m"], ["в1", "о1", 1], ["в2", "о2", 0]]


def _load_main():
    spec = importlib.util.spec_from_file_location("adapter_main", MODULE_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["adapter_main"] = module
    spec.loader.exec_module(module)
    return module


def _run_main(tmp_path, monkeypatch, responses, rows=None, *, run_context=None,
              package_name="CI09000001_test", validation=()):
    package = make_package(
        tmp_path, {"Лист1": {"rows": rows or ROWS}}, name=package_name, validation=validation
    )
    main = _load_main()
    client = FakeClient(responses)
    monkeypatch.setattr(main, "LlmClient", lambda config, out_dir: client)
    return main.main(
        validation_report=package / "validation_report.docx",
        test_set=package / "test_set.xlsx",
        development_report=package / "development_report.docx",
        assessor_instruction=package / "assessor_instruction.txt",
        run_context=run_context,
    )


def test_main_uses_run_context_agent_ci_as_basket_id(tmp_path, monkeypatch):
    ports = _run_main(
        tmp_path, monkeypatch, [layout_answer(), metric_answer()],
        run_context={"agent_ci": "CI86242115"}, package_name="корзина_агента_23",
    )
    assert ports["monitoring_metric"]["basket_id"] == "CI86242115"
    assert ports["km_result"]["basket_id"] == "CI86242115"


def test_monitoring_metric_carries_consumer_fields(tmp_path, monkeypatch):
    ports = _run_main(tmp_path, monkeypatch, [layout_answer(), metric_answer()])
    metric = ports["monitoring_metric"]
    assert metric["contract_version"] == "laim-monitoring-metric.v2"
    assert metric["umr_version"] == "laim-umr.v2"
    assert metric["status"] == "computed"
    assert metric["score_column"] == "main_metric"
    assert metric["assessment_mode"] == "qa"
    assert metric["primary_validation"]["affects_monitoring"] is False
    source = metric["scoring"]["sources"][0]
    assert set(source) == {"source_id", "column_name", "role", "normalization",
                            "polarity"}
    assert metric["scoring"]["missing_policy"] == "exclude_unit"
    assert metric["baseline"]["scale"] in ("ratio", "raw")
    assert isinstance(metric["baseline"]["value"], float)
    assert isinstance(metric["baseline"]["recomputed_value"], float)
    assert metric["baseline"]["reconciliation"] == "match"
    assert metric["aggregation"]["method"] in ("mean", "frequency_weighted_mean")


def test_km_result_is_run_report(tmp_path, monkeypatch):
    ports = _run_main(tmp_path, monkeypatch, [layout_answer(), metric_answer()])
    report = ports["km_result"]
    assert report["contract_version"] == "laim-run-report.v1"
    assert report["stages"] and report["input_sha256"]
    assert report["llm"]["calls"] >= 2
    assert report["km"]["value"] == 0.5
    assert Path(ports["umr_artifact"]).exists()


def test_not_computable_still_publishes_umr(tmp_path, monkeypatch):
    bad = metric_answer(sources=[{"column_id": "ZZ", "role": "final_score",
                                   "normalization": "numeric",
                                   "polarity": "direct"}])
    ports = _run_main(tmp_path, monkeypatch, [layout_answer()] + [bad] * 3)
    assert ports["monitoring_metric"]["status"] == "not_computable"
    assert len(ports["reference_umr"]) == 2
    assert ports["km_result"]["status"] == "not_computable"


def test_official_baseline_missing_reason_code(tmp_path, monkeypatch):
    ports = _run_main(tmp_path, monkeypatch, [
        layout_answer(),
        metric_answer(reported_value={"state": "not_declared", "value": None,
                                       "raw": None})])
    metric = ports["monitoring_metric"]
    assert metric["status"] == "not_computable"
    assert metric["reason_code"] == "official_baseline_missing"
    assert metric["baseline"]["recomputed_value"] == 0.5
    # km_result обязан согласоваться с monitoring_metric, а не рапортовать успех.
    assert ports["km_result"]["status"] == "not_computable"
    assert any(w["code"] == "official_baseline_missing"
               for w in ports["km_result"]["warnings"])


def test_descriptor_source_files_match_disk():
    import json
    descriptor = json.loads((MODULE_DIR / "descriptor.json").read_text("utf-8"))
    listed = set(descriptor["script"]["runConfiguration"]["sourceFiles"])
    actual = {
        str(path.relative_to(MODULE_DIR))
        for path in MODULE_DIR.rglob("*.py")
        if "__pycache__" not in path.parts
        and path.relative_to(MODULE_DIR).parts[0] != "tests"
    }
    assert listed == actual
    ports = {port["name"]: port for port in descriptor["ports"]}
    assert ports["run_context"]["in"] is True and ports["run_context"]["required"] is False


def test_main_logs_basket_error_details_before_raising(tmp_path, monkeypatch, caplog):
    """Детали BasketError попадают в лог платформы: иначе прод — чёрный ящик."""
    package = make_package(tmp_path, {"Лист1": {"rows": ROWS}})
    main = _load_main()
    monkeypatch.setattr(main, "LlmClient", lambda config, out_dir: object())

    def failing_run_package(path, out_dir, client, sheet_name, run_context):
        raise LayoutError(
            "Канонический UMR не прошёл валидацию",
            missing_required_values={"input_query": {"count": 2,
                                                       "source_rows": [365, 366]}},
        )

    monkeypatch.setattr(main, "run_package", failing_run_package)
    with caplog.at_level("ERROR"), pytest.raises(LayoutError):
        main.main(
            validation_report=package / "validation_report.docx",
            test_set=package / "test_set.xlsx",
            development_report=package / "development_report.docx",
            assessor_instruction=package / "assessor_instruction.txt",
        )
    assert "source_rows" in caplog.text and "366" in caplog.text


def test_package_name_keeps_cyrillic_identity(tmp_path):
    node = _load_main()
    basket = tmp_path / "Корзина агента вкладов" / "тестовая корзина.xlsx"
    basket.parent.mkdir()
    basket.touch()

    name = node._package_name(basket)

    assert "basket" not in name and "агента" in name and " " not in name


@pytest.mark.parametrize("scale", ["ratio", "percent"])
def test_small_percent_publishes_same_ratio_baseline(tmp_path, monkeypatch, scale):
    proposal = metric_answer(scale=scale, reported_value={
        "state": "declared", "value": None, "raw": "0.9%"})
    ports = _run_main(
        tmp_path, monkeypatch, [layout_answer(), proposal],
        rows=[["q", "a", "m"], ["вопрос", "ответ", 0.009]],
        validation=("Ключевая метрика Accuracy равна 0.9%",))
    baseline = ports["monitoring_metric"]["baseline"]
    assert baseline["scale"] == "ratio"
    assert baseline["value"] == pytest.approx(0.009)
    assert baseline["recomputed_value"] == pytest.approx(0.009)
    assert baseline["reconciliation"] == "match"
