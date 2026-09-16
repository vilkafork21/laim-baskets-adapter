"""Проверка корпуса: живой шлюз или явно обозначенное воспроизведение плана.

Манифест и артефакты хранятся вне публичного репозитория. Никаких загрузок
корпуса, исполнения pickle или подмены отсутствующих документов здесь нет.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

NODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NODE_ROOT))

from laim_basket.config import llm_config  # noqa: E402
from laim_basket.contract import monitoring_metric  # noqa: E402
from laim_basket.llm.client import LlmClient  # noqa: E402
from laim_basket.pipeline import run_package  # noqa: E402


class ReviewedPlanClient:
    """Не модель: возвращает проверенные человеком/аудитором планы из манифеста."""

    def __init__(self, proposals: dict):
        self.proposals = proposals
        self.config = SimpleNamespace(model="reviewed-plan-replay-NOT-LLM")
        self.structured_output = None
        self.calls = self.repair_turns = self.transport_retries = 0

    def chat(self, messages, label, response_schema=None):
        self.calls += 1
        stage = label.split("_turn")[0].split("_chunk")[0]
        proposal = self.proposals[stage]
        if stage == "baseline":
            # Номера pNNN должны присутствовать именно в текущем chunk.
            text = messages[-1]["content"]
            proposal = [c for c in proposal if c["paragraph"] + ":" in text]
        return json.dumps(proposal, ensure_ascii=False)


def _json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _safe_input(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Нет файла внутри корня корпуса: {relative}")
    return path


def run_case(case: dict, root: Path, out: Path, mode: str, model: str) -> dict:
    """Каждая попытка изолирована; отказ одного агента не прерывает корпус."""
    case_id = case["case_id"]
    if not case_id or Path(case_id).name != case_id or case_id in {".", ".."}:
        raise ValueError("case_id должен быть одним безопасным именем каталога")
    target = out / case_id
    target.mkdir(parents=True, exist_ok=True)
    record = {
        "case_id": case_id,
        "category": case.get("category", "real"),
        "execution_mode": mode,
        "model": model if mode == "live" else "NOT-LLM",
        "provenance_note": case.get("provenance_note", ""),
        "input_sha256": {},
        "checks": {},
    }
    started = time.monotonic()
    log = logging.FileHandler(target / "run.log", mode="w", encoding="utf-8")
    logging.getLogger().addHandler(log)
    try:
        with tempfile.TemporaryDirectory(prefix="laim-corpus-") as work:
            package = Path(work) / case_id
            package.mkdir()
            missing = []
            for port in (
                "test_set",
                "validation_report",
                "development_report",
                "assessor_instruction",
            ):
                relative = case.get("inputs", {}).get(port)
                if not relative:
                    missing.append(port)
                    continue
                path = _safe_input(root, relative)
                record["input_sha256"][port] = hashlib.sha256(path.read_bytes()).hexdigest()
                suffix = path.suffix.lower()
                if port == "test_set" and suffix not in {".xlsx", ".xlsm"}:
                    raise ValueError("Порт test_set принимает OOXML, а не Parquet/pickle")
                shutil.copyfile(path, package / (port + suffix))
            if missing:
                record.update(status="incomplete_inputs", missing_ports=missing)
            else:
                config = replace(llm_config(), model=model) if mode == "live" else None
                client = (
                    LlmClient(config, target / "llm")
                    if mode == "live"
                    else ReviewedPlanClient(case["reviewed_proposals"])
                )
                result = run_package(
                    package,
                    target / "output",
                    client=client,
                    sheet_name=case.get("sheet_name", ""),
                    agent_ci=case.get("agent_ci", ""),
                )
                metric = monitoring_metric(result)
                _json(target / "monitoring_metric.json", metric)
                score = result.km.get("main_metric") or {}
                record.update(
                    status=result.status,
                    reason_code=result.km.get("reason_code"),
                    reason=result.km.get("reason"),
                    sheet=result.report["decisions"]["sheet"],
                    mode=result.report["decisions"].get("assessment_mode"),
                    umr_rows=len(result.umr.frame),
                    umr_columns=list(result.umr.frame),
                    official=score.get("value"),
                    recomputed=score.get("recomputed_value"),
                    reconciliation=result.km["reconciliation"]["status"],
                    coverage=result.km.get("coverage"),
                    baseline_state=result.report["baseline"]["state"],
                    baseline_value=result.report["baseline"]["value"],
                    dropped_rows=result.report["dropped_rows"],
                    warnings=result.report["warnings"],
                    calls=client.calls,
                )
                expected = case.get("expected", {})
                for key, value in expected.items():
                    if key == "total_units":
                        observed = (record.get("coverage") or {}).get("total_units")
                    else:
                        observed = record.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        okay = observed is not None and abs(observed - value) <= 1e-9
                    else:
                        okay = observed == value
                    record["checks"][key] = {
                        "expected": value,
                        "observed": observed,
                        "passed": okay,
                    }
                record["expected_checks_passed"] = all(
                    c["passed"] for c in record["checks"].values()
                )
    except Exception as exc:  # изоляция корпуса, не подавление ошибок внутри ноды
        logging.exception("Случай %s не завершён", case_id)
        record.update(
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
            reason_code=getattr(exc, "reason_code", None),
            details=getattr(exc, "details", {}),
        )
    finally:
        logging.getLogger().removeHandler(log)
        log.close()
    record["seconds"] = round(time.monotonic() - started, 3)
    _json(target / "result.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("reviewed", "live"), default="reviewed")
    parser.add_argument("--model", default="gigachat-3-ultra")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    cases = manifest["cases"]
    identifiers = [case["case_id"] for case in cases]
    if len(identifiers) != len(set(identifiers)):
        parser.error("Повтор case_id в манифесте")
    results = []
    for case in cases:
        result = run_case(case, args.root, args.out, args.mode, args.model)
        results.append(result)
        print(
            f"{result['case_id']}: {result['status']}; "
            f"official={result.get('official')}; recomputed={result.get('recomputed')}"
        )
        _json(args.out / "results.json", results)
    from collections import Counter

    summary = {
        "execution_mode": args.mode,
        "model": args.model if args.mode == "live" else "NOT-LLM",
        "cases": len(results),
        "statuses": dict(Counter(r["status"] for r in results)),
        "expected_failures": sum(r.get("expected_checks_passed") is False for r in results),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    }
    _json(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(
        any(r["status"] == "error" or r.get("expected_checks_passed") is False for r in results)
    )


if __name__ == "__main__":
    raise SystemExit(main())
