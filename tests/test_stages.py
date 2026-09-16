"""Регрессии изоляции L/S/K и четырёхпортового контракта."""

import json
from copy import deepcopy

import pytest
from conftest import layout_answer, metric_answer, source
from helpers import FakeClient, make_package

from laim_basket.errors import LlmError, PackageError
from laim_basket.llm.prompts import baseline_messages
from laim_basket.pipeline import run_package


def score(**fields):
    answer = metric_answer(**fields)
    for key in ("reported_value", "threshold", "comparator"):
        answer.pop(key, None)
    return answer


def baseline(raw="0.5", paragraph="p002", **fields):
    return [
        dict(
            metric_name="Accuracy",
            raw=raw,
            paragraph=paragraph,
            kind="value",
            slice_label=None,
            is_key_metric=True,
            **fields,
        )
    ]


def package(tmp_path):
    return make_package(
        tmp_path,
        {
            "Данные": {"rows": [["q", "a", "m"], ["в1", "о1", 1], ["в2", "о2", 0]]},
            "Описание": {"rows": [["name", "info"], ["q", "вопрос"]]},
        },
    )


def test_all_stages_computed_on_data_plus_description(tmp_path):
    client = FakeClient([layout_answer(sheet_name="Данные"), score(), baseline()])
    result = run_package(package(tmp_path), tmp_path / "out", client=client)
    assert result.status == "computed"
    assert client.labels == ["layout_turn1", "metric_turn1", "baseline_turn1"]
    assert result.report["baseline"]["value"] == 0.5
    assert [x["stage"] for x in result.report["stages"]] == ["read", "layout", "metric", "baseline"]


@pytest.mark.parametrize(
    "k_response,k_state,k_reason",
    [
        ([], "not_declared", "official_baseline_missing"),
        (baseline("0.5") + baseline("0.8", "p003"), "ambiguous", "ambiguous_baseline"),
    ],
)
@pytest.mark.parametrize("s_ok", [True, False])
def test_failures_never_switch_sheet_and_scores_survive_k_failure(
    tmp_path, s_ok, k_response, k_state, k_reason
):
    root = make_package(
        tmp_path,
        {
            "Данные": {"rows": [["q", "a", "m"], ["в1", "о1", 1], ["в2", "о2", 0]]},
            "Описание": {"rows": [["name", "info"], ["q", "вопрос"]]},
        },
        validation=("Accuracy 0.5", "Accuracy 0.8"),
    )
    bad = score(sources=[source("ZZ", "final_score")])
    client = FakeClient(
        [layout_answer(sheet_name="Данные")] + ([score()] if s_ok else [bad] * 3) + [k_response]
    )
    result = run_package(root, tmp_path / "out", client=client)
    assert result.status == "not_computable"
    assert result.km["reason_code"] == (k_reason if s_ok else "score_plan_failed")
    assert result.report["baseline"]["state"] == k_state
    assert result.report["decisions"]["sheet"] == "Данные"
    assert result.umr.frame["input_query"].tolist() == ["в1", "в2"]
    assert ("main_metric" in result.umr.frame) == s_ok
    assert sum(x.startswith("layout_") for x in client.labels) == 1
    assert all(x in result.km["reason"] for x in ("Этап", "Ожидалось", "Получено", "Действие"))


def test_score_failure_does_not_hide_official_value(tmp_path):
    bad = score(sources=[source("ZZ", "final_score")])
    client = FakeClient([layout_answer(sheet_name="Данные")] + [bad] * 3 + [baseline()])
    result = run_package(package(tmp_path), tmp_path / "out", client=client)
    assert result.km["reason_code"] == "score_plan_failed"
    assert result.report["baseline"]["state"] == "declared"
    assert result.report["baseline"]["value"] == 0.5
    assert "main_metric" not in result.umr.frame


@pytest.mark.parametrize("fail_label", ["metric", "baseline"])
def test_transport_failure_after_layout_is_not_fatal(tmp_path, fail_label):
    class FailingClient(FakeClient):
        def chat(self, messages, label, response_schema=None):
            if label.startswith(fail_label + "_"):
                self.labels.append(label)
                raise LlmError("Контурный шлюз недоступен")
            return super().chat(messages, label, response_schema)

    responses = [layout_answer(sheet_name="Данные")]
    responses += [baseline()] if fail_label == "metric" else [score()]
    client = FailingClient(responses)
    result = run_package(package(tmp_path), tmp_path / "out", client=client)
    assert result.status == "not_computable" and len(result.umr.frame) == 2
    assert result.km["reason_code"] == (
        "score_plan_failed" if fail_label == "metric" else "official_baseline_missing"
    )


def test_baseline_prompt_cannot_see_development_or_scores():
    report = {
        "port": "validation_report",
        "name": "validation_report.docx",
        "paragraphs": ("Accuracy 0.5",),
    }
    messages = baseline_messages(report, ["Данные", "Описание"], "Accuracy")
    user = messages[-1]["content"]
    assert "p001: Accuracy 0.5" in user
    assert "Данные" in user and "Описание" in user
    assert "development_report" not in user and "assessor_instruction" not in user
    assert "column_inventory" not in user and "recomputed_value" not in user


def test_agent_ci_setting_overrides_package_identity(tmp_path):
    client = FakeClient([layout_answer(sheet_name="Данные"), score(), baseline()])
    result = run_package(package(tmp_path), tmp_path / "out", client=client, agent_ci="ci09774440")
    assert result.report["basket_id"] == "CI09774440"
    assert result.measurement_plan.basket_id == "CI09774440"
    assert result.excel_name == "umr_CI09774440.xlsx"


@pytest.mark.parametrize("agent_ci", ["../escape", "foo", "CI123/456"])
def test_bad_agent_ci_setting_is_rejected_without_llm(tmp_path, agent_ci):
    client = FakeClient([])
    with pytest.raises(PackageError):
        run_package(package(tmp_path), tmp_path / "out", client=client, agent_ci=agent_ci)
    assert client.calls == 0


def test_public_descriptor_changes_only_optional_agent_ci_and_sources():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    # Тест не зависит от git checkout: зафиксирован снимок исходного descriptor.
    original = json.loads((root / "tests" / "base_descriptor.json").read_text())
    current = json.loads((root / "descriptor.json").read_text())
    clean = deepcopy(current)
    components = clean["ui"]["settings"][0]["components"][0]["config"]["components"]
    setting = components.pop()
    assert setting["parameter"] == "agent_ci" and setting["defaultValue"] == ""

    for added in (
        "laim_basket/metric/baseline.py",
        "laim_basket/metric/nonadditive.py",
        "laim_basket/context.py",
        "laim_basket/contract.py",
    ):
        clean["script"]["runConfiguration"]["sourceFiles"].remove(added)
    assert clean == original


def _formula_package(tmp_path, formula, cached=True):
    import xml.etree.ElementTree as ET
    import zipfile

    root = make_package(
        tmp_path,
        {
            "Данные": {
                "rows": [
                    ["q", "a", "score", "x", "y"],
                    ["в1", "о1", formula, 1, 1],
                    ["в2", "о2", "=SUM(D3:E3)/2", 0, 0],
                ]
            }
        },
    )
    if cached:
        path = root / "test_set.xlsx"
        with zipfile.ZipFile(path) as z:
            files = {name: z.read(name) for name in z.namelist()}
        ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        sheet = ET.fromstring(files["xl/worksheets/sheet1.xml"])
        for cell in sheet.iter(ns + "c"):
            if cell.get("r") in {"C2", "C3"}:
                value = cell.find(ns + "v")
                if value is None:
                    value = ET.SubElement(cell, ns + "v")
                value.text = "1" if cell.get("r") == "C2" else "0"
        files["xl/worksheets/sheet1.xml"] = ET.tostring(sheet)
        with zipfile.ZipFile(path, "w") as z:
            for name, contents in files.items():
                z.writestr(name, contents)
    return root


def test_cached_row_formula_is_a_real_score_source(tmp_path):
    client = FakeClient([layout_answer(sheet_name="Данные"), score(), baseline()])
    result = run_package(
        _formula_package(tmp_path, "=SUM(D2:E2)/2"), tmp_path / "out", client=client
    )
    assert result.status == "computed"
    assert result.umr.frame["main_metric"].tolist() == [1.0, 0.0]
    assert any(w["code"] == "formula_score_cached" for w in result.report["warnings"])


@pytest.mark.parametrize("formula,cached", [("=D2+D3", True), ("=SUM(D2:E2)/2", False)])
def test_nonlocal_or_uncached_formula_does_not_publish_fabricated_score(tmp_path, formula, cached):
    client = FakeClient([layout_answer(sheet_name="Данные")] + [score()] * 3 + [baseline()])
    result = run_package(
        _formula_package(tmp_path, formula, cached), tmp_path / "out", client=client
    )
    assert result.km["reason_code"] == "score_plan_failed"
    assert result.report["baseline"]["state"] == "declared"
    assert "main_metric" not in result.umr.frame


def test_k_bad_schema_is_repaired_without_rerunning_s(tmp_path):
    client = FakeClient(
        [layout_answer(sheet_name="Данные"), score(), {"state": "ambiguous"}, baseline()]
    )
    result = run_package(package(tmp_path), tmp_path / "out", client=client)
    assert result.status == "computed"
    assert client.labels == ["layout_turn1", "metric_turn1", "baseline_turn1", "baseline_turn2"]


def test_k_exhausted_schema_repair_preserves_scores(tmp_path):
    client = FakeClient(
        [layout_answer(sheet_name="Данные"), score()] + [{"state": "ambiguous"}] * 3
    )
    result = run_package(package(tmp_path), tmp_path / "out", client=client)
    assert result.km["reason_code"] == "official_baseline_missing"
    assert result.umr.frame["main_metric"].tolist() == [1.0, 0.0]
    assert client.labels.count("metric_turn1") == 1


def test_agent_ci_setting_reaches_platform_outputs(tmp_path, monkeypatch):
    import main

    root = package(tmp_path)
    client = FakeClient([layout_answer(sheet_name="Данные"), score(), baseline()])
    monkeypatch.setattr(main, "LlmClient", lambda *args: client)
    outputs = main.main(
        root / "validation_report.docx",
        root / "test_set.xlsx",
        root / "development_report.docx",
        root / "assessor_instruction.txt",
        agent_ci="CI09774440",
    )
    assert outputs["monitoring_metric"]["basket_id"] == "CI09774440"
    assert outputs["km_result"]["basket_id"] == "CI09774440"


def test_k_ambiguity_is_preserved_in_monitoring_metric(tmp_path, monkeypatch):
    import main

    root = make_package(
        tmp_path,
        {"Данные": {"rows": [["q", "a", "m"], ["в1", "о1", 1]]}},
        validation=("Accuracy 0.5", "Accuracy 0.8"),
    )
    client = FakeClient(
        [layout_answer(sheet_name="Данные"), score(), baseline() + baseline("0.8", "p003")]
    )
    monkeypatch.setattr(main, "LlmClient", lambda *args: client)
    outputs = main.main(
        root / "validation_report.docx",
        root / "test_set.xlsx",
        root / "development_report.docx",
        root / "assessor_instruction.txt",
    )
    assert outputs["monitoring_metric"]["reason_code"] == "ambiguous_baseline"
    assert outputs["monitoring_metric"]["baseline"]["recomputed_value"] == 1.0
    assert outputs["km_result"]["baseline"]["state"] == "ambiguous"
