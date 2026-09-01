"""Единственный продовый путь: пакет → УМР → MeasurementPlan → КМ."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path

from .config import llm_config
from .errors import BasketError, LayoutError, MeasurementPlanError, NotEvaluableError, StructuredOutputError
from .evidence.sheet_evidence import build_sheet_evidence
from .export.umr import export_umr_workbook
from .hashing import sha256_canonical_json, sha256_dataframe
from .llm.client import LlmClient
from .llm.generate import generate_layout, generate_measurement_plan
from .models import RunContext, RunResult
from .publish import publish_umr
from .reading.docx_reader import read_document_spans
from .reading.package_scan import scan_package
from .reading.xlsx_reader import read_workbook

_AUXILIARY_TXT_NAMES = {"distrib.txt"}


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    _json(temporary, payload)
    os.replace(temporary, path)


def reset_out(out_dir: str | Path) -> None:
    out = Path(out_dir)
    shutil.rmtree(out / "debug", ignore_errors=True)
    for pattern in ("umr_*.xlsx", "km.json", "failure.json", ".*.tmp", ".*.tmp.xlsx"):
        for stale in out.glob(pattern):
            stale.unlink(missing_ok=True)


def _basket_id(package_name: str) -> str:
    match = re.search(r"(?i)ci[0-9]+", package_name)
    return match.group(0).upper() if match else package_name


def load_run_context(input_path: str | Path) -> RunContext:
    manifest = scan_package(input_path)
    baskets = manifest["baskets"]
    documents = manifest["documents"]
    files_by_name = {item["name"]: item for item in manifest["files"]}
    if len(baskets) != 1:
        raise LayoutError("Пакет должен содержать ровно одну XLSX-корзину", found=baskets)
    document_kinds = [files_by_name[file_name]["kind"] for file_name in documents]
    unexpected_txt = [
        item
        for item in manifest["files"]
        if item["kind"] == "document_txt"
        and item["name"] not in documents
        and item["name"].casefold() not in _AUXILIARY_TXT_NAMES
    ]
    if (
        len(documents) != 3
        or document_kinds.count("document_txt") > 1
        or unexpected_txt
    ):
        raise LayoutError(
            "Пакет должен содержать два DOCX-отчёта и инструкцию DOCX либо "
            "UTF-8 TXT с каноническим именем assessor_instruction.txt",
            found=[files_by_name[file_name] for file_name in documents],
            unexpected_txt=unexpected_txt,
        )
    package_dir = Path(manifest["package_dir"])
    basket_path = package_dir / baskets[0]
    loaded_documents = []
    for index, file_name in enumerate(sorted(documents), start=1):
        document_id = f"doc-{index}"
        loaded_documents.append({
            "id": document_id,
            "file_name": file_name,
            "sha256": files_by_name[file_name]["sha256"],
            "spans": read_document_spans(
                package_dir / file_name,
                document_id,
                files_by_name[file_name]["kind"],
            ),
        })
    return RunContext(
        basket_id=_basket_id(manifest["package_name"]),
        package_dir=package_dir,
        basket_path=basket_path,
        file_hashes={item["name"]: item["sha256"] for item in manifest["files"]},
        sheets=read_workbook(basket_path),
        documents=tuple(loaded_documents),
    )


def _column_inventory(context, frame, layout) -> list[dict[str, object]]:
    result = []
    sheet = context.sheets[layout.sheet_name]
    for column_id, name in layout.column_names.items():
        values = frame[name].tolist()
        present = [value for value in values if value is not None and str(value).strip()]
        column = layout.region.column_letters.index(column_id)
        vertically_merged = any(
            row2 > row1
            and column1 <= column <= column2
            and row2 >= layout.first_data_row - 1
            and row1 <= layout.last_data_row - 1
            for row1, column1, row2, column2 in sheet.merged
        )
        result.append({
            "column_id": column_id,
            "canonical_raw_name": name,
            "non_null": len(present),
            "samples": [str(value)[:120] for value in present[:3]],
            "formula_rows": list(layout.formula_rows.get(column_id, ())),
            "vertically_merged": vertically_merged,
        })
    return result


def _not_evaluable(context: RunContext, reason: BasketError) -> dict[str, object]:
    return {
        "report_version": "laim-km.v2",
        "status": "not_evaluable",
        "value_source": None,
        "recomputed_value": None,
        "coverage": None,
        "reconciliation": {"status": "not_evaluable", "difference": None},
        "evidence": {},
        "threshold_verdict": None,
        "reason": str(reason),
        "reason_code": reason.reason_code,
        "details": reason.details,
        "main_metric": None,
        "basket_id": context.basket_id,
    }


def run_package(
    input_path: str | Path,
    out_dir: str | Path,
    llm_preset: str = "openrouter",
    client=None,
    sheet_name: str = "",
) -> RunResult:
    """sheet_name задаёт обязательный лист корзины (книги с листами нескольких
    агентов различимы только оператором); пустое значение — автоопределение."""
    root = Path(out_dir)
    reset_out(root)
    debug = root / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    context = load_run_context(input_path)
    _json(debug / "run_context.json", {
        "basket_id": context.basket_id,
        "basket": context.basket_path.name,
        "file_hashes": context.file_hashes,
        "documents": [
            {"id": document["id"], "file_name": document["file_name"], "spans": len(document["spans"])}
            for document in context.documents
        ],
    })
    llm = client or LlmClient(llm_config(llm_preset), debug)
    workbook_evidence = {
        "sheets": [build_sheet_evidence(sheet) for sheet in context.sheets.values()]
    }
    # Одна книга может нести листы разных агентов (классификация и диалоги в
    # одном файле). Пригодность листа доказывает построенный план, а не выбор
    # модели: лист, на котором КМ не строится, запрещается и разбор повторяется.
    plan = None
    rejected_sheets: list[str] = []
    while True:
        layout_proposal, layout, frame, conversion = generate_layout(
            llm, context, workbook_evidence, tuple(rejected_sheets), sheet_name
        )
        _json(debug / "layout_proposal.json", layout_proposal)
        _json(debug / "resolved_layout.json", layout.to_dict())
        if conversion["umr_validation"]["status"] != "passed":
            raise LayoutError("UMR validation blocked publication", validation=conversion["umr_validation"])
        _json(debug / "conversion.json", conversion)
        try:
            plan_proposal, plan, published, km = generate_measurement_plan(
                llm, context, layout, frame, _column_inventory(context, frame, layout)
            )
            _json(debug / "measurement_proposal.json", plan_proposal)
            _json(debug / "measurement_plan.json", plan.to_dict())
            break
        except (NotEvaluableError, MeasurementPlanError, StructuredOutputError) as exc:
            rejected_sheets.append(layout.sheet_name)
            # Заданный оператором лист не перебирается: неудача плана на нём —
            # честный not_evaluable, а не попытка чужого листа.
            if sheet_name or len(rejected_sheets) >= len(context.sheets):
                published, km = publish_umr(frame, layout, None), _not_evaluable(context, exc)
                break
            _json(debug / f"rejected_sheet_{len(rejected_sheets)}.json", {
                "sheet_name": layout.sheet_name, "reason": str(exc),
                "reason_code": exc.reason_code, "details": exc.details,
            })
    _json(debug / "publication.json", published.to_dict())

    excel_name = f"umr_{context.basket_id}.xlsx"
    temporary_excel = root / f".{excel_name}.{uuid.uuid4().hex}.tmp.xlsx"
    root.mkdir(parents=True, exist_ok=True)
    try:
        export_summary = export_umr_workbook(published, temporary_excel)
        os.replace(temporary_excel, root / excel_name)
    finally:
        temporary_excel.unlink(missing_ok=True)

    provenance = {
        "input_sha256": context.file_hashes,
        "canonical_sha256": sha256_dataframe(published.frame),
        "layout_sha256": sha256_canonical_json(layout.to_dict()),
        "measurement_plan_sha256": sha256_canonical_json(plan.to_dict()) if plan else None,
    }
    km.update({
        "basket_id": context.basket_id,
        "files": {"excel": excel_name},
        "provenance": provenance,
    })
    _atomic_json(root / "km.json", km)
    summary = {
        "status": km["status"],
        "excel": excel_name,
        "km": "km.json",
        "rows": len(published.frame),
        "export": export_summary,
    }
    _json(debug / "run_summary.json", summary)
    return RunResult(
        status=km["status"],
        umr=published,
        km=km,
        excel_name=excel_name,
        measurement_plan=plan,
    )


def write_failure(out_dir: str | Path, stage: str, error: BasketError) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "failure.json", {"status": "invalid_input", "stage": stage, **error.to_dict()})
