"""Одна Sber DS-нода: четыре unstructured-входа -> UMR и контракт КМ."""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import laim_monitoring

from laim_basket.config import llm_config
from laim_basket.errors import BasketError, PackageError
from laim_basket.llm.client import LlmClient
from laim_basket.models import RunResult
from laim_basket.pipeline import run_package
from laim_basket.reading.package_scan import classify_file

logger = logging.getLogger(__name__)

_PORT_FILES = {
    "test_set": {"basket_xlsx": "test_set.xlsx"},
    "assessor_instruction": {
        "document_docx": "assessor_instruction.docx",
        "document_txt": "assessor_instruction.txt",
    },
    "development_report": {"document_docx": "development_report.docx"},
    "validation_report": {"document_docx": "validation_report.docx"},
}


def _port_files(value: str | Path, port_name: str) -> list[Path]:
    if value is None:
        raise PackageError(f"Порт {port_name} обязателен", port=port_name)
    path = Path(value)
    if not path.exists():
        raise PackageError(
            f"Путь из порта {port_name} не существует: {path}",
            port=port_name,
            path=str(path),
        )
    if path.is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and not any(part.startswith((".", "~$")) for part in candidate.relative_to(path).parts)
    )


def _resolve_artifact(value: str | Path, port_name: str) -> tuple[Path, str]:
    """Единственный допустимый artifact порта и его каноническое имя в пакете."""
    expected_files = _PORT_FILES[port_name]
    classified = [(path, classify_file(path)) for path in _port_files(value, port_name)]
    matches = [(path, expected_files[kind]) for path, kind in classified if kind in expected_files]
    if len(matches) != 1:
        raise PackageError(
            f"Порт {port_name} должен содержать ровно один допустимый artifact",
            port=port_name,
            expected_kinds=sorted(expected_files),
            found=[{"path": str(path), "kind": kind} for path, kind in classified],
        )
    return matches[0]


def _package_name(test_set: Path) -> str:
    identity = f"{test_set.parent.name}_{test_set.stem}"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", identity).strip("-._")
    return safe or "basket"


def _prepare_package(root: Path, artifacts: dict[str, tuple[Path, str]]) -> Path:
    package = root / _package_name(artifacts["test_set"][0])
    package.mkdir(parents=True)
    for source, target_name in artifacts.values():
        shutil.copy2(source, package / target_name)
    return package


def _normalized(value: object, scale: str) -> tuple[float, str]:
    number = Decimal(str(value))
    if scale == "percent":
        return float(number / Decimal(100)), "ratio"
    if scale in {"ratio", "raw"}:
        return float(number), scale
    raise PackageError("MeasurementPlan содержит неизвестную шкалу", scale=scale)


def _monitoring_metric(result: RunResult) -> dict[str, object]:
    """Порт monitoring_metric: формула, входы, единица оценки и доказанный baseline."""
    plan = result.measurement_plan
    contract = {
        "contract_version": laim_monitoring.VERSION,
        "laim_monitoring_version": laim_monitoring.__version__,
        "basket_id": result.km.get("basket_id"),
    }
    if plan is None or result.status != "computed":
        return {
            **contract,
            "status": "not_computable",
            "reason": result.km.get("reason", "КМ не вычислена"),
            "reason_code": result.km.get("reason_code"),
        }
    recomputed_value, scale = _normalized(result.km["recomputed_value"], plan.scale)
    if plan.reported_value is None:
        return {
            **contract,
            "status": "not_computable",
            "reason": "Validation report не содержит официальный baseline",
            "reason_code": "official_baseline_missing",
            "recomputed_value": recomputed_value,
        }
    if result.km["reconciliation"]["status"] != "match":
        # Инвариант: _reconciliation_gate не пропускает computed без совпадения.
        raise PackageError("computed-план без воспроизведённого baseline", reconciliation=result.km["reconciliation"])
    baseline_value, _ = _normalized(plan.reported_value, plan.scale)
    return {
        **contract,
        "status": "computed",
        "metric_name": plan.metric_name,
        "assessment_mode": plan.assessment_mode,
        "formula": plan.formula,
        "inputs": [
            {
                "name": item["name"],
                "column": result.umr.published_columns[item["column_id"]],
                "judged": item["judged"],
            }
            for item in plan.inputs
        ],
        "baseline": {
            "value": baseline_value,
            "recomputed_value": recomputed_value,
            "reconciliation": "match",
            "reported_value": float(plan.reported_value),
            "reported_scale": plan.scale,
            "scale": scale,
        },
        "primary_validation": {
            "threshold": float(plan.threshold) if plan.threshold is not None else None,
            "comparator": plan.comparator,
            "verdict": result.km.get("threshold_verdict"),
        },
        "evidence": {key: list(value) for key, value in plan.evidence.items()},
    }


def _parquet_safe(frame):
    """Object-колонки со смешанными типами приводятся к строкам: Arrow-порт
    SberDS не принимает колонки, где соседствуют int и str."""
    result = frame.copy()
    for column in result.columns:
        values = result[column]
        if values.dtype != object:
            continue
        kinds = {
            type(item).__name__
            for item in values
            if item is not None and item == item
        }
        if len(kinds) > 1:
            result[column] = [
                None if item is None or item != item else str(item)
                for item in values
            ]
    return result


def main(
    validation_report: str | Path,
    test_set: str | Path,
    development_report: str | Path,
    assessor_instruction: str | Path,
    model_id: str = "minimax-m2.5",
    sheet_name: str = "",
):
    """Запустить полный laim-basket внутри одной Sber DS-ноды."""
    values = {
        "validation_report": validation_report,
        "test_set": test_set,
        "development_report": development_report,
        "assessor_instruction": assessor_instruction,
    }
    artifacts = {
        name: _resolve_artifact(value, name)
        for name, value in values.items()
    }
    work_dir = Path(tempfile.mkdtemp(prefix="laim-basket-sberds-"))
    package = _prepare_package(work_dir, artifacts)
    out_dir = work_dir / "result"
    config = llm_config("contour")
    if model_id:
        config = replace(config, model=model_id)
    client = LlmClient(config, out_dir / "debug")
    try:
        result = run_package(
            package, out_dir, llm_preset="contour", client=client, sheet_name=sheet_name
        )
    except BasketError as exc:
        # Платформа показывает только str(exc), а debug-каталог гибнет вместе
        # с временной папкой контейнера — детали обязаны попасть в лог.
        logger.error(
            "Конвейер упал (%s): %s | детали: %s",
            exc.reason_code, exc,
            json.dumps(exc.details, ensure_ascii=False, default=str)[:4000],
        )
        raise
    return {
        "reference_umr": _parquet_safe(result.umr.frame),
        "monitoring_metric": _monitoring_metric(result),
        "km_result": result.km,
        "umr_artifact": str(out_dir / result.excel_name),
    }
