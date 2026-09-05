"""Физика разметки: границы, роли, группировка.

Проверяется исполнимость маппинга и форма данных — не честность модели.
Границами данных и режимом оценки владеет только код.
"""
from __future__ import annotations

import math

import jsonschema
from openpyxl.utils import column_index_from_string

from .errors import LayoutError
from .llm.schemas import LAYOUT_SCHEMA
from .models import ResolvedLayout
from .reading.formulas import vertical_aggregate_cells
from .reading.xlsx_reader import RawSheet
from .transform.canon import raw_column_names, reference_answer_names
from .transform.values import blank
from .transform.region import build_region


def _column_index(address: str, width: int, path: str) -> int:
    try:
        index = column_index_from_string(address) - 1
    except ValueError as exc:
        raise LayoutError(f"{path}: некорректный адрес колонки", address=address) from exc
    if index >= width:
        raise LayoutError(f"{path}: колонка вне листа", column=address, width=width)
    return index


def _addresses(proposal: dict) -> list[tuple[str, str]]:
    """Все адреса предложения парой (путь, буква колонки)."""
    roles = proposal["roles"]
    result = [("roles.input_query", roles["input_query"])]
    output = roles["output_answer"]
    if isinstance(output, dict):
        result.extend(
            (f"roles.output_answer.coalesce[{index}]", address)
            for index, address in enumerate(output["coalesce"])
        )
    elif output is not None:
        result.append(("roles.output_answer", output))
    for name in ("query_id", "session_id", "scenario", "assessor_id"):
        if roles[name] is not None:
            result.append((f"roles.{name}", roles[name]))
    result.extend(
        (f"roles.reference_answers[{index}]", address)
        for index, address in enumerate(roles["reference_answers"])
    )
    if proposal["grouping"]["column"] is not None:
        result.append(("grouping.column", proposal["grouping"]["column"]))
    if proposal["dialogue_blob"] is not None:
        result.append(("dialogue_blob.column", proposal["dialogue_blob"]["column"]))
    if proposal["weight_column"] is not None:
        result.append(("weight", proposal["weight_column"]))
    return result


def _materialize(proposal: dict, sheet: RawSheet, region) -> tuple[dict, dict, dict | None, dict | None]:
    """Адреса предложения -> внутренние роли на именах колонок региона."""

    def name_of(address: str, path: str) -> str:
        return region.columns[_column_index(address, sheet.n_cols, path)]

    source_roles = proposal["roles"]
    output = source_roles["output_answer"]
    if isinstance(output, dict):
        resolved_output: dict | None = {
            "coalesce": [
                name_of(address, f"roles.output_answer.coalesce[{index}]")
                for index, address in enumerate(output["coalesce"])
            ]
        }
    elif output is not None:
        resolved_output = {"source": name_of(output, "roles.output_answer")}
    else:
        resolved_output = None

    def role_or_none(name: str) -> dict | None:
        value = source_roles[name]
        return {"source": name_of(value, f"roles.{name}")} if value is not None else None

    roles: dict[str, object] = {
        "input_query": {"source": name_of(source_roles["input_query"], "roles.input_query")},
        "output_answer": resolved_output,
        "query_id": role_or_none("query_id") or "synthesize",
        "session_id": role_or_none("session_id"),
        "scenario": role_or_none("scenario"),
        "assessor_id": role_or_none("assessor_id"),
        "reference_answers": [
            name_of(address, f"roles.reference_answers[{index}]")
            for index, address in enumerate(source_roles["reference_answers"])
        ],
    }
    grouping = {"kind": proposal["grouping"]["kind"], "column": None}
    if proposal["grouping"]["column"] is not None:
        grouping["column"] = name_of(proposal["grouping"]["column"], "grouping.column")
    blob = None
    if proposal["dialogue_blob"] is not None:
        blob = dict(proposal["dialogue_blob"])
        blob["column"] = name_of(blob["column"], "dialogue_blob.column")
    weight = None
    if proposal["weight_column"] is not None:
        weight = {"source": name_of(proposal["weight_column"], "weight"),
                  "column_id": proposal["weight_column"]}
    return roles, grouping, blob, weight


def _prove_roles(roles: dict, grouping: dict, blob: dict | None) -> None:
    kind = grouping["kind"]
    if kind == "blob_row":
        input_source = roles["input_query"]["source"]
        blob_source = blob["column"] if blob is not None else None
        if blob is None or input_source != blob_source:
            raise LayoutError(
                "blob_row требует общий источник input_query и dialogue_blob",
                input_query_source=input_source, dialogue_blob_source=blob_source,
            )
        if roles["output_answer"] is not None:
            raise LayoutError(
                "blob_row получает output_answer из blob, роль должна быть null",
            )
    elif blob is not None:
        raise LayoutError(
            "dialogue_blob допустим только при grouping=blob_row",
            actual_grouping_kind=kind,
        )

    claimed: dict[str, str] = {}
    for name in ("input_query", "query_id", "session_id", "scenario", "assessor_id"):
        value = roles.get(name)
        if not isinstance(value, dict) or "source" not in value:
            continue
        source = value["source"]
        if source in claimed:
            raise LayoutError("Одна колонка получила две канонические роли",
                              column=source, roles=[claimed[source], name])
        claimed[source] = name
    output = roles["output_answer"]
    if isinstance(output, dict):
        output_sources = output.get("coalesce") or [output["source"]]
        for source in output_sources:
            if source in claimed:
                raise LayoutError("Одна колонка получила две канонические роли",
                                  column=source, roles=[claimed[source], "output_answer"])
            claimed[source] = "output_answer"
    for index, source in enumerate(roles["reference_answers"]):
        if source in claimed:
            raise LayoutError("Колонка reference_answer уже занята ролью", column=source)
        claimed[source] = f"reference_answers[{index}]"

    session_role = roles.get("session_id")
    if (kind == "column" and isinstance(session_role, dict)
            and session_role["source"] != grouping["column"]):
        raise LayoutError("roles.session_id и grouping.column должны совпадать")
    if (kind == "column" and grouping["column"] in claimed
            and claimed[grouping["column"]] not in {"query_id", "session_id"}):
        grouping.update(kind="none", column=None)
        kind = "none"
    if kind == "column" and grouping["column"] is None:
        raise LayoutError("grouping=column требует column")


def _present_numbers(region, source: str) -> list | None:
    """Заполненные значения колонки, если все они числовые, иначе None."""
    values = [row[region.columns.index(source)] for row in region.rows]
    present = [value for value in values if not blank(value)]
    numeric = all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in present
    )
    return present if present and numeric else None


def _prove_references(region, roles: dict) -> None:
    """Идентификатор разметчика не должен подменяться бинарным голосом."""
    assessor = roles.get("assessor_id")
    if isinstance(assessor, dict):
        present = _present_numbers(region, assessor["source"])
        if present is not None and all(float(value) in (0.0, 1.0) for value in present):
            raise LayoutError(
                "Колонка assessor_id содержит только значения 0/1 — это голос "
                "разметчика, а не его идентификатор",
                column=assessor["source"],
                repair_hint="назначь assessor_id=null; колонки голосов назначает план метрики",
            )


def _prove_weight(sheet: RawSheet, region, weight: dict | None) -> None:
    if weight is None:
        return
    source = weight["source"]
    index = region.columns.index(source)
    invalid = []
    for source_row, row in zip(region.source_rows, region.rows):
        value = row[index]
        if blank(value):
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
            or (source_row - 1, index) in sheet.formulas
        ):
            invalid.append({"row": source_row, "value": str(value)[:80]})
    if invalid:
        raise LayoutError(
            "Weight должен содержать только положительные целые значения без Excel-формул",
            column=source, invalid=invalid[:10],
        )


def _data_bounds(sheet: RawSheet, proposal: dict, first_data0: int) -> tuple[int, int]:
    """Границы данных по непустым input_query, merge и формульным итогам."""
    query_index = _column_index(proposal["roles"]["input_query"], sheet.n_cols,
                                "roles.input_query")
    # Строка с вертикальным агрегатом в любой колонке — футер таблицы, даже
    # если под запросами стоит метка «ИТОГО».
    footer_rows = {row for row, _column in vertical_aggregate_cells(sheet.formulas)}
    candidates = [
        row for row in range(first_data0, sheet.n_rows)
        if not blank(sheet.grid[row][query_index]) and row not in footer_rows
    ]
    for row1, col1, row2, _col2 in sheet.merged:
        if row1 >= first_data0 and col1 <= query_index <= _col2 and not blank(sheet.grid[row1][col1]):
            candidates.append(row2)
    if not candidates:
        raise LayoutError("В колонке input_query нет строк данных")
    return query_index, min(max(candidates), sheet.n_rows - 1)


def resolve_layout(proposal: dict, sheets: dict[str, RawSheet], basket_id: str,
                   pinned_sheet: str, rejected_sheets: frozenset[str]) -> ResolvedLayout:
    try:
        jsonschema.validate(proposal, LAYOUT_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise LayoutError(f"Layout не соответствует схеме: {exc.message}",
                          path=exc.json_path) from exc
    sheet_name = proposal["sheet_name"]
    if pinned_sheet and sheet_name != pinned_sheet:
        raise LayoutError("Оператор закрепил лист — выбери именно его",
                          pinned=pinned_sheet, actual=sheet_name)
    if sheet_name in rejected_sheets:
        raise LayoutError("Лист уже отвергнут, выбери другой",
                          rejected=sorted(rejected_sheets),
                          available=sorted(set(sheets) - rejected_sheets))
    if sheet_name not in sheets:
        raise LayoutError("Выбран неизвестный лист", sheet=sheet_name,
                          available=list(sheets))
    sheet = sheets[sheet_name]
    if proposal["grouping"]["kind"] != "column":
        # Для остальных видов группировки поле не имеет смысла: игнорировать
        # безопаснее, чем ронять валидный прогон из-за совещательного шума.
        proposal["grouping"]["column"] = None

    header_rows = list(proposal["header_rows"])
    if header_rows != sorted(header_rows) or any(
        right != left + 1 for left, right in zip(header_rows, header_rows[1:])
    ):
        raise LayoutError("Строки заголовка должны быть последовательными и упорядоченными")
    if header_rows[-1] >= sheet.n_rows:
        raise LayoutError("После заголовка нет строк данных", header_rows=header_rows)
    first_data0 = header_rows[-1]

    query_index, last_data0 = _data_bounds(sheet, proposal, first_data0)
    vertical_data_merge = any(
        row2 > row1 and row2 >= first_data0 and row1 <= last_data0
        for row1, _c1, row2, _c2 in sheet.merged
    )
    vertical_query_merge = any(
        row2 > row1 and row2 >= first_data0 and row1 <= last_data0
        and col1 <= query_index <= col2
        for row1, col1, row2, col2 in sheet.merged
    )
    if vertical_query_merge or (
        proposal["grouping"]["kind"] == "none" and vertical_data_merge
    ):
        proposal["grouping"] = {"kind": "merged_rows", "column": None}

    region = build_region(sheet, header_rows, last_data0 + 1)
    roles, grouping, blob, weight = _materialize(proposal, sheet, region)
    _prove_roles(roles, grouping, blob)
    _prove_references(region, roles)
    _prove_weight(sheet, region, weight)
    if grouping["kind"] == "merged_rows" and not vertical_data_merge:
        raise LayoutError("grouping=merged_rows не подтвержден vertical merge")

    canonical = {
        "source_row_id", "query_id", "session_id", "input_query_count",
        "input_query", "output_answer", "reference_group_id", "turn_index",
    }
    if roles["scenario"] is not None:
        canonical.add("scenario")
    if roles["assessor_id"] is not None:
        canonical.add("assessor_id")
    canonical.update(reference_answer_names(len(roles["reference_answers"])))
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
    evidence = {
        "sheet": sheet_name,
        "header_rows": ",".join(str(row) for row in header_rows),
        **{path: address for path, address in _addresses(proposal)},
    }
    return ResolvedLayout(
        basket_id=basket_id,
        sheet_name=sheet_name,
        header_rows=tuple(header_rows),
        first_data_row=first_data0 + 1,
        last_data_row=last_data0 + 1,
        roles=roles,
        grouping=grouping,
        dialogue_blob=blob,
        weight=weight,
        evidence=evidence,
        column_names=column_names,
        formula_rows=formula_rows,
        region=region,
    )
