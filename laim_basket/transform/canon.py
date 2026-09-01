"""Сборка аудируемых строк УМР с сохранением каждой физической колонки-источника."""

import pandas as pd

from ..umr_schema import validate_flat_canon
from .grouping import GroupedTable
from .identity import build_identity
from .region import TableRegion


def raw_column_names(columns: list[str], reserved: set[str]) -> dict[str, str]:
    used = set(reserved)
    result = {}
    for name in columns:
        candidate = name
        while candidate in used:
            candidate += "__raw"
        result[name] = candidate
        used.add(candidate)
    return result


def _column(table: GroupedTable, source: str) -> list[object]:
    index = table.columns.index(source)
    return [row[index] for row in table.rows]


def _text(value: object) -> str | None:
    return None if value is None or str(value).strip() == "" else str(value)


def _scalar(value: object) -> object | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def build_canon(
    grouped: GroupedTable,
    region: TableRegion,
    config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    roles = config["roles"]
    identity = build_identity(
        grouped.rows,
        grouped.columns,
        roles,
        config.get("weight"),
        groups=grouped.group_index,
        query_id_override=grouped.query_id_override,
    )
    data: dict[str, list[object]] = {
        "source_row_id": grouped.source_rows,
        "query_id": identity.query_id,
        "session_id": identity.session_id,
        "input_query_count": identity.input_query_count,
    }
    if config["grouping"]["kind"] == "blob_row":
        data["input_query"] = [_text(value) for value in _column(grouped, "__blob_input_query")]
        data["output_answer"] = [_text(value) for value in _column(grouped, "__blob_output_answer")]
    else:
        data["input_query"] = [_text(value) for value in _column(grouped, roles["input_query"]["source"])]
        output = roles.get("output_answer")
        if isinstance(output, dict) and "coalesce" in output:
            candidates = [
                [_text(value) for value in _column(grouped, source)]
                for source in output["coalesce"]
            ]
            data["output_answer"] = [
                next((values[row] for values in candidates if values[row] is not None), None)
                for row in range(len(grouped.rows))
            ]
        else:
            data["output_answer"] = (
                [_text(value) for value in _column(grouped, output["source"])]
                if output else [None] * len(grouped.rows)
            )
    scenario = roles.get("scenario")
    if isinstance(scenario, dict) and "source" in scenario:
        data["scenario"] = [_scalar(value) for value in _column(grouped, scenario["source"])]
    assessor = roles.get("assessor_id")
    if assessor:
        data["assessor_id"] = [_scalar(value) for value in _column(grouped, assessor["source"])]
    for index, source in enumerate(roles.get("reference_answers", [])):
        name = "reference_answer" if index == 0 else f"reference_answer_{index + 1}"
        data[name] = [_text(value) for value in _column(grouped, source)]
    if grouped.group_index is not None:
        data["reference_group_id"] = [f"group-{value}" for value in grouped.group_index]
        counters: dict[int, int] = {}
        turn_index = []
        for group in grouped.group_index:
            counters[group] = counters.get(group, 0) + 1
            turn_index.append(counters[group])
        data["turn_index"] = turn_index

    canonical = set(data)
    raw_names = raw_column_names(
        region.columns,
        canonical | {"reference_group_id", "turn_index"},
    )
    frame = pd.DataFrame(data, dtype=object)
    for source in region.columns:
        frame[raw_names[source]] = _column(grouped, source)
    validation = validate_flat_canon(
        frame,
        canonical,
        session_scoped=(
            grouped.group_index is not None or isinstance(roles.get("session_id"), dict)
        ),
    )
    report = {
        "row_accounting": {**region.accounting, **grouped.accounting, "canon_rows": len(frame)},
        "raw_columns": raw_names,
        "identity": identity.accounting,
        "grouping": {"kind": config["grouping"]["kind"], **grouped.accounting},
        "umr_validation": validation,
    }
    return frame, report
