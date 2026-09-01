"""Журнал: события уходят и в logging, и в отчёт порта."""
from __future__ import annotations


from laim_basket.journal import Journal


def test_report_carries_contract_and_events(tmp_path, caplog):
    journal = Journal()
    with caplog.at_level("INFO"):
        journal.stage("layout", "ok", ms=120)
        journal.decision(sheet="Лист1", assessment_mode="qa")
        journal.warning("reconciliation_mismatch", "пересчёт 0.5, заявлено 0.9")
        journal.dropped("blank_input_query", [3, 7])
        journal.set_inputs({"test_set.xlsx": "a" * 64})
        journal.set_llm(model="glm-5.2", structured_output=True,
                        calls=2, repair_turns=1, transport_retries=0)
    report = journal.report(basket_id="CI09000001", status="computed",
                            km={"name": "Accuracy", "value": 0.9})
    assert report["contract_version"] == "laim-run-report.v1"
    assert report["stages"] == [{"stage": "layout", "ms": 120, "outcome": "ok"}]
    assert report["decisions"]["sheet"] == "Лист1"
    assert report["warnings"][0]["code"] == "reconciliation_mismatch"
    assert report["dropped_rows"]["blank_input_query"] == [3, 7]
    assert "test_set.xlsx" in report["input_sha256"]
    assert report["llm"]["structured_output"] is True
    assert any("layout" in record.message for record in caplog.records)
    assert any(record.levelname == "WARNING" for record in caplog.records)
