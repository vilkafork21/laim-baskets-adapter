"""Инвентарь входных файлов без исполнения pickle, макросов и Excel-формул."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
import sys
from collections import Counter
from pathlib import Path

NODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NODE_ROOT))

from laim_basket.reading.docx_reader import read_document_paragraphs  # noqa: E402
from laim_basket.reading.xlsx_reader import read_workbook  # noqa: E402


def inspect_file(path: Path, root: Path) -> dict:
    record = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    try:
        suffix = path.suffix.casefold()
        if suffix in {".xlsx", ".xlsm"}:
            sheets = read_workbook(path)
            record.update(
                status="workbook_read",
                sheets=[
                    {
                        "name": sheet.name,
                        "rows": sheet.n_rows,
                        "columns": sheet.n_cols,
                        "formulas": len(sheet.formulas),
                        "merges": len(sheet.merged),
                        "empty_formula_cache": sum(
                            sheet.grid[r][c] is None for r, c in sheet.formulas
                        ),
                    }
                    for sheet in sheets.values()
                ],
            )
        elif suffix == ".docx":
            paragraphs = read_document_paragraphs(path, "document_docx")
            record.update(
                status="document_read",
                paragraphs=len(paragraphs),
                characters=sum(map(len, paragraphs)),
            )
        elif suffix == ".parquet":
            import pyarrow.parquet as pq  # инструмент аудита, не runtime-зависимость

            file = pq.ParquetFile(path)
            count = sum(batch.num_rows for batch in file.iter_batches(batch_size=2048))
            if count != file.metadata.num_rows:
                raise ValueError("Число прочитанных строк не совпало с metadata")
            record.update(status="parquet_fully_read", rows=count, columns=file.schema.names)
        elif suffix == ".pkl":
            # genops разбирает байткод, но НЕ вызывает ни GLOBAL, ни REDUCE.
            with path.open("rb") as stream:
                operations = sum(1 for _ in pickletools.genops(stream))
            record.update(
                status="pickle_static_only",
                opcodes=operations,
                limitation="Содержимое объектов не десериализовано и не выполнено",
            )
        else:
            text = path.read_text(encoding="utf-8-sig")
            if suffix == ".jsonl":
                for line in text.splitlines():
                    if line.strip():
                        json.loads(line)
            record.update(status="text_read", characters=len(text))
    except Exception as exc:  # один дефект не скрывает остальные файлы
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = [
        inspect_file(p, args.root)
        for p in sorted(args.root.rglob("*"))
        if p.is_file() and "__MACOSX" not in p.parts and not p.name.startswith(".")
    ]
    payload = {
        "files": len(records),
        "unique_sha256": len({r["sha256"] for r in records}),
        "statuses": dict(Counter(r["status"] for r in records)),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2))
    return int(any(r["status"] == "error" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
