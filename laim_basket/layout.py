"""Проверка физического предложения по книге и доказательство сохранения строк."""

from __future__ import annotations

import math

import jsonschema
from openpyxl.utils import column_index_from_string, get_column_letter

from .contracts import LAYOUT_SCHEMA
from .errors import LayoutError
from .models import ResolvedLayout, RunContext
from .reading.formulas import vertical_aggregate_cells
from .reading.headers import header_merge_map, merge_header
from .transform.canon import raw_column_names
from .transform.region import build_region
from .transform.values import normalize_key


def _present(value: object) -> bool:
    return not (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or str(value).strip() == ""
    )


def _derived_summary_row(sheet, row: int, query_index: int, aggregate_cells: set[tuple[int, int]]) -> bool:
    """Формульный итоговый хвост учитывается, но наблюдением не является."""
    return not _present(sheet.grid[row][query_index]) and any(
        (row, column) in aggregate_cells for column in range(sheet.n_cols)
    )


def _prove_weight(sheet, region, weight: dict[str, str] | None) -> None:
    if weight is None:
        return
    source = weight["source"]
    index = region.columns.index(source)
    raw_index = region.raw_index(source)
    invalid = []
    for source_row, row in zip(region.source_rows, region.rows):
        value = row[index]
        if not _present(value):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = math.nan
        if (
            isinstance(value, bool)
            or not math.isfinite(number)
            or number < 1
            or not number.is_integer()
            or (source_row - 1, raw_index) in sheet.formulas
        ):
            invalid.append({"row": source_row, "value": str(value)[:80]})
    if invalid:
        raise LayoutError(
            "Weight должен содержать только положительные целые значения без Excel-формул",
            column=source,
            invalid=invalid[:10],
        )


def _address_index(address: dict[str, object], width: int, path: str) -> int:
    try:
        index = column_index_from_string(str(address["column"])) - 1
    except ValueError as exc:
        raise LayoutError(f"{path}: некорректный column_id", address=address) from exc
    if index >= width:
        raise LayoutError(f"{path}: колонка вне листа", column=address["column"], width=width)
    return index


def _prove_additional_header_rows(
    sheet,
    header_rows: list[int],
    addresses: list[tuple[str, dict[str, object]]],
) -> None:
    """Строка данных не может стать второй строкой шапки без merge-структуры."""
    for row1 in header_rows[1:]:
        row0 = row1 - 1
        previous_rows0 = {value - 1 for value in header_rows if value < row1}
        for path, address in addresses:
            column = _address_index(address, sheet.n_cols, path)
            value = sheet.grid[row0][column]
            if not _present(value):
                continue
            supported = any(
                column1 <= column <= column2
                and column2 > column1
                and any(range_row1 <= previous <= range_row2 for previous in previous_rows0)
                for range_row1, column1, range_row2, column2 in sheet.merged
            )
            if supported:
                continue
            raise LayoutError(
                "Дополнительная строка шапки содержит значение канонической роли "
                "без родительского horizontal merge и похожа на первую строку данных",
                row=row1,
                column=address["column"],
                role_path=path,
                value=str(value)[:120],
                repair_hint=(
                    "Убери эту строку из header_rows. Несколько строк шапки допустимы, "
                    "только когда дополнительный уровень физически подтвержден "
                    "горизонтальным merge предыдущего уровня."
                ),
            )


def _addresses(proposal: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
    roles = proposal["roles"]
    result = [("roles.input_query", roles["input_query"])]
    output = roles["output_answer"]
    if isinstance(output, dict) and "coalesce" in output:
        result.extend(
            (f"roles.output_answer.coalesce[{index}]", address)
            for index, address in enumerate(output["coalesce"])
        )
    elif output is not None:
        result.append(("roles.output_answer", output))
    if roles.get("session_id") is not None:
        result.append(("roles.session_id", roles["session_id"]))
    if roles["assessor_id"] is not None:
        result.append(("roles.assessor_id", roles["assessor_id"]))
    if isinstance(roles["query_id"], dict):
        result.append(("roles.query_id", roles["query_id"]))
    if roles["scenario"] is not None:
        result.append(("roles.scenario", roles["scenario"]))
    result.extend(
        (f"roles.reference_answers[{index}]", address)
        for index, address in enumerate(roles["reference_answers"])
    )
    grouping = proposal["grouping"]
    if grouping["column"] is not None:
        result.append(("grouping.column", grouping["column"]))
    if proposal["dialogue_blob"] is not None:
        result.append(("dialogue_blob.column", proposal["dialogue_blob"]["column"]))
    if proposal["weight"] is not None:
        result.append(("weight", proposal["weight"]))
    return result


def _resolved_name(address: dict[str, object], path: str, sheet, region, header_rows: list[int]) -> str:
    raw_index = _address_index(address, sheet.n_cols, path)
    header_rows0 = [row - 1 for row in header_rows]
    merge_map = header_merge_map(sheet, header_rows0)
    visible = [merge_header(sheet, [row], raw_index, merge_map) for row in header_rows0]
    echo = address["header"]
    if echo is None:
        if any(visible):
            raise LayoutError(f"{path}: header=null для непустого заголовка", visible=visible)
    elif normalize_key(echo) not in {normalize_key(value) for value in visible if value}:
        combined = region.columns[region.raw_indexes.index(raw_index)]
        if normalize_key(echo) != normalize_key(combined):
            raise LayoutError(
                f"{path}: эхо заголовка не совпадает с физической колонкой",
                column=address["column"], echo=echo, visible=visible,
            )
    return region.columns[region.raw_indexes.index(raw_index)]


def _materialize(proposal: dict[str, object], sheet, region) -> tuple[dict, dict, dict | None, dict | None]:
    header_rows = proposal["header_rows"]
    source_roles = proposal["roles"]

    def resolve(address: dict[str, object], path: str) -> dict[str, str]:
        return {"source": _resolved_name(address, path, sheet, region, header_rows)}

    output = source_roles["output_answer"]
    if isinstance(output, dict) and "coalesce" in output:
        resolved_output = {
            "coalesce": [
                resolve(address, f"roles.output_answer.coalesce[{index}]")["source"]
                for index, address in enumerate(output["coalesce"])
            ]
        }
    else:
        resolved_output = (
            resolve(output, "roles.output_answer") if output is not None else None
        )

    roles: dict[str, object] = {
        "input_query": resolve(source_roles["input_query"], "roles.input_query"),
        "output_answer": resolved_output,
        "query_id": (
            resolve(source_roles["query_id"], "roles.query_id")
            if isinstance(source_roles["query_id"], dict) else "synthesize"
        ),
        "session_id": (
            resolve(source_roles["session_id"], "roles.session_id")
            if source_roles.get("session_id") is not None else None
        ),
        "scenario": source_roles["scenario"],
        "assessor_id": (
            resolve(source_roles["assessor_id"], "roles.assessor_id")
            if source_roles["assessor_id"] is not None else None
        ),
        "reference_answers": [
            _resolved_name(address, f"roles.reference_answers[{index}]", sheet, region, header_rows)
            for index, address in enumerate(source_roles["reference_answers"])
        ],
    }
    if source_roles["scenario"] is not None:
        roles["scenario"] = resolve(source_roles["scenario"], "roles.scenario")

    grouping_source = proposal["grouping"]
    grouping = {"kind": grouping_source["kind"], "column": None}
    if grouping_source["column"] is not None:
        grouping["column"] = _resolved_name(
            grouping_source["column"], "grouping.column", sheet, region, header_rows
        )
    blob = None
    if proposal["dialogue_blob"] is not None:
        blob = dict(proposal["dialogue_blob"])
        blob["column"] = _resolved_name(
            blob["column"], "dialogue_blob.column", sheet, region, header_rows
        )
    weight = None
    if proposal["weight"] is not None:
        weight = resolve(proposal["weight"], "weight")
        weight["column_id"] = proposal["weight"]["column"]
    return roles, grouping, blob, weight


def _prove_roles(roles: dict[str, object], grouping: dict, blob: dict | None) -> None:
    kind = grouping["kind"]
    if kind == "blob_row":
        input_source = roles["input_query"]["source"]
        blob_source = blob["column"] if blob is not None else None
        output = roles["output_answer"]
        output_source = (
            output.get("source") or output.get("coalesce")
            if isinstance(output, dict) else None
        )
        details = {
            "actual": {
                "grouping_kind": "blob_row",
                "input_query_source": input_source,
                "dialogue_blob_source": blob_source,
                "output_answer_source": output_source,
            },
            "required_for_blob_row": {
                "common_input_and_dialogue_blob_source": True,
                "output_answer": None,
            },
            "flat_rows_repair": {
                "when": "input_query и output_answer находятся в отдельных колонках строки",
                "grouping_kind": ["column", "none"],
                "dialogue_blob": None,
            },
        }
        if blob is None or input_source != blob_source:
            raise LayoutError(
                "blob_row требует общий источник input_query и dialogue_blob",
                **details,
            )
        if output is not None:
            raise LayoutError(
                "blob_row получает output_answer из blob, роль должна быть null",
                **details,
            )
    elif blob is not None:
        raise LayoutError(
            "dialogue_blob допустим только при grouping=blob_row",
            actual_grouping_kind=kind,
            valid_repairs={
                "flat_rows": {
                    "grouping_kind": ["column", "none"],
                    "dialogue_blob": None,
                },
                "blob_row": {
                    "grouping_kind": "blob_row",
                    "input_query_source": blob["column"],
                    "output_answer": None,
                },
            },
        )

    claimed: dict[str, str] = {}
    for name in ("input_query", "query_id", "session_id", "scenario", "assessor_id"):
        value = roles.get(name)
        if not isinstance(value, dict) or "source" not in value:
            continue
        source = value["source"]
        if source in claimed:
            raise LayoutError("Одна колонка получила две канонические роли", column=source, roles=[claimed[source], name])
        claimed[source] = name
    output = roles.get("output_answer")
    output_sources = (
        output["coalesce"]
        if isinstance(output, dict) and "coalesce" in output
        else [output["source"]]
        if isinstance(output, dict) and "source" in output
        else []
    )
    for index, source in enumerate(output_sources):
        role = f"output_answer.coalesce[{index}]" if len(output_sources) > 1 else "output_answer"
        if source in claimed:
            raise LayoutError(
                "Одна колонка получила две канонические роли",
                column=source,
                roles=[claimed[source], role],
            )
        claimed[source] = role
    for index, source in enumerate(roles["reference_answers"]):
        if source in claimed:
            raise LayoutError("Колонка reference_answer уже занята ролью", column=source)
        claimed[source] = f"reference_answers[{index}]"
    session_role = roles.get("session_id")
    if (
        kind == "column"
        and isinstance(session_role, dict)
        and session_role["source"] != grouping["column"]
    ):
        raise LayoutError("roles.session_id и grouping.column должны совпадать")
    if (
        kind == "column"
        and grouping["column"] in claimed
        and claimed[grouping["column"]] not in {"query_id", "session_id"}
    ):
        grouping.update(kind="none", column=None)
        kind = "none"
    if kind == "column" and grouping["column"] is None:
        raise LayoutError("grouping=column требует column")
    if kind != "column" and grouping["column"] is not None:
        raise LayoutError("grouping.column допустим только для grouping=column")


def resolve_layout(proposal: dict[str, object], context: RunContext) -> ResolvedLayout:
    try:
        jsonschema.validate(proposal, LAYOUT_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise LayoutError(f"Layout не соответствует схеме: {exc.message}", path=exc.json_path) from exc
    proposal.pop("first_data_row", None)
    proposal.pop("last_data_row", None)
    proposal.setdefault("evidence", {})
    if proposal["grouping"]["kind"] != "column":
        # Для остальных видов группировки поле не имеет смысла. Игнорировать его
        # безопаснее, чем ронять валидный прогон из-за совещательного шума модели.
        proposal["grouping"]["column"] = None
    if proposal["basket_id"] != context.basket_id:
        raise LayoutError("basket_id layout не совпадает с пакетом")
    sheet_name = proposal["sheet_name"]
    if sheet_name not in context.sheets:
        raise LayoutError("Выбран неизвестный лист", sheet=sheet_name, available=list(context.sheets))
    expected_ignored = set(context.sheets) - {sheet_name}
    if set(proposal["ignored_sheets"]) != expected_ignored:
        raise LayoutError(
            "Каждый невыбранный лист должен быть явно учтен",
            expected=sorted(expected_ignored), actual=sorted(proposal["ignored_sheets"]),
        )
    sheet = context.sheets[sheet_name]
    header_rows = list(proposal["header_rows"])
    if header_rows != sorted(header_rows) or any(
        right != left + 1 for left, right in zip(header_rows, header_rows[1:])
    ):
        raise LayoutError("Строки заголовка должны быть последовательными и упорядоченными")
    if header_rows[-1] >= sheet.n_rows:
        raise LayoutError("После заголовка нет строк данных", header_rows=header_rows)
    first_data0 = header_rows[-1]
    addresses = _addresses(proposal)
    raw_indexes = {_address_index(address, sheet.n_cols, path) for path, address in addresses}
    _prove_additional_header_rows(sheet, header_rows, addresses)
    query_index = _address_index(proposal["roles"]["input_query"], sheet.n_cols, "roles.input_query")
    aggregate_cells = vertical_aggregate_cells(sheet.formulas)
    candidates = [
        row for row in range(first_data0, sheet.n_rows)
        if _present(sheet.grid[row][query_index]) and (row, query_index) not in aggregate_cells
    ]
    for row1, col1, row2, col2 in sheet.merged:
        if row1 >= first_data0 and col1 <= query_index <= col2 and _present(sheet.grid[row1][col1]):
            candidates.append(row2)
    if not candidates:
        raise LayoutError("В колонке input_query нет строк данных")
    last_data0 = min(max(candidates), sheet.n_rows - 1)
    vertical_data_merge = any(
        row2 > row1 and row2 >= first_data0 and row1 <= last_data0
        for row1, _column1, row2, _column2 in sheet.merged
    )
    vertical_query_merge = any(
        row2 > row1 and row2 >= first_data0 and row1 <= last_data0
        and column1 <= query_index <= column2
        for row1, column1, row2, column2 in sheet.merged
    )
    if vertical_query_merge or (
        proposal["grouping"]["kind"] == "none" and vertical_data_merge
    ):
        proposal["grouping"] = {"kind": "merged_rows", "column": None}

    first_header0 = header_rows[0] - 1
    for row in range(first_header0):
        populated = sum(_present(sheet.grid[row][column]) for column in raw_indexes)
        if populated >= 2:
            raise LayoutError(
                "Layout отрезает похожую на данные строку перед заголовком",
                row=row + 1, populated_mapped_columns=populated,
            )
    for row in range(last_data0 + 1, sheet.n_rows):
        if _derived_summary_row(sheet, row, query_index, aggregate_cells):
            continue
        unsafe = [
            get_column_letter(column + 1) for column in raw_indexes
            if _present(sheet.grid[row][column]) and (row, column) not in aggregate_cells
        ]
        if unsafe:
            raise LayoutError(
                "Layout оставляет данные используемых колонок за границей таблицы",
                row=row + 1, columns=unsafe,
            )

    region = build_region(sheet, header_rows, last_data0 + 1)
    roles, grouping, blob, weight = _materialize(proposal, sheet, region)
    _prove_roles(roles, grouping, blob)
    _prove_weight(sheet, region, weight)
    if grouping["kind"] == "merged_rows" and not vertical_data_merge:
        raise LayoutError("grouping=merged_rows не подтвержден vertical merge")

    row_ledger = []
    header_set = {row - 1 for row in header_rows}
    for row, values in enumerate(sheet.grid):
        nonempty = sum(_present(value) for value in values)
        if not nonempty:
            continue
        if row in header_set:
            role = "header"
        elif first_data0 <= row <= last_data0:
            role = "data"
        elif row < first_header0:
            role = "preamble"
        elif _derived_summary_row(sheet, row, query_index, aggregate_cells):
            role = "derived_summary"
        else:
            role = "metadata"
        row_ledger.append({"row": row + 1, "role": role, "nonempty_cells": nonempty})

    canonical = {
        "source_row_id", "query_id", "session_id", "input_query_count",
        "input_query", "output_answer", "reference_group_id", "turn_index",
    }
    if roles.get("scenario") is not None:
        canonical.add("scenario")
    if roles.get("assessor_id") is not None:
        canonical.add("assessor_id")
    if roles["reference_answers"]:
        canonical.add("reference_answer")
        canonical.update(
            f"reference_answer_{index}"
            for index in range(2, len(roles["reference_answers"]) + 1)
        )
    raw_names = raw_column_names(region.columns, canonical)
    column_names = {
        letter: raw_names[name]
        for letter, name in zip(region.column_letters, region.columns)
    }
    formula_rows: dict[str, tuple[int, ...]] = {}
    for column, letter in enumerate(region.column_letters):
        rows = tuple(
            row + 1 for row in range(first_data0, last_data0 + 1)
            if (row, column) in sheet.formulas
        )
        if rows:
            formula_rows[letter] = rows
    physical_evidence = {
        "sheet": sheet_name,
        "header_rows": ",".join(str(row) for row in header_rows),
        **{
            path: f"{address['column']}:{address.get('header') or ''}"
            for path, address in addresses
        },
    }
    return ResolvedLayout(
        basket_id=context.basket_id,
        sheet_name=sheet_name,
        ignored_sheets=tuple(proposal["ignored_sheets"]),
        header_rows=tuple(header_rows),
        first_data_row=first_data0 + 1,
        last_data_row=last_data0 + 1,
        roles=roles,
        grouping=grouping,
        dialogue_blob=blob,
        weight=weight,
        evidence=physical_evidence,
        column_names=column_names,
        formula_rows=formula_rows,
        row_ledger=tuple(row_ledger),
        region=region,
    )
