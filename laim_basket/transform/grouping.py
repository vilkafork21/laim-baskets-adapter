"""Детерминированная группировка строк и развёртка диалог-блоба без потерь."""

import ast
import re
from dataclasses import dataclass, field

from ..errors import LayoutError
from ..reading.xlsx_reader import RawSheet
from .region import TableRegion


@dataclass
class GroupedTable:
    columns: list[str]
    rows: list[list[object]]
    source_rows: list[int]
    group_index: list[int] | None
    query_id_override: list[str] | None = None
    accounting: dict[str, object] = field(default_factory=dict)


class _DisjointSets:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _number_groups(roots: list[int]) -> list[int]:
    order: dict[int, int] = {}
    for root in roots:
        order.setdefault(root, len(order))
    return [order[root] for root in roots]


def _merged_rows(sheet: RawSheet, region: TableRegion) -> tuple[list[list[object]], list[int], int]:
    sets = _DisjointSets(len(region.rows))
    rows = [list(row) for row in region.rows]
    raw_to_region = {raw: index for index, raw in enumerate(region.raw_indexes)}
    filled = 0
    first, last = region.first_data_row0, region.first_data_row0 + len(rows) - 1
    for row1, col1, row2, col2 in sheet.merged:
        indexes = list(range(max(row1, first) - first, min(row2, last) - first + 1))
        if len(indexes) < 2:
            continue
        for index in indexes[1:]:
            sets.union(indexes[0], index)
        for raw_column in range(col1, col2 + 1):
            if raw_column not in raw_to_region:
                continue
            column = raw_to_region[raw_column]
            top = rows[indexes[0]][column]
            for index in indexes[1:]:
                if rows[index][column] is None and top is not None:
                    rows[index][column] = top
                    filled += 1
    groups = _number_groups([sets.find(index) for index in range(len(rows))])
    return rows, groups, filled


def _column_groups(region: TableRegion, column: str) -> list[int]:
    index = region.columns.index(column)
    keys: dict[str, int] = {}
    groups, previous = [], None
    for source_row, row in zip(region.source_rows, region.rows):
        value = row[index]
        key = str(value).strip() if value is not None and str(value).strip() else previous
        if key is None:
            raise LayoutError("Пустая первая группа", row=source_row, column=column)
        previous = key
        keys.setdefault(key, len(keys))
        groups.append(keys[key])
    return groups


def _decode_python_item(text: str) -> str | None:
    for quote in ('"""', "'''"):
        if quote in text:
            continue
        try:
            decoded = ast.literal_eval(f"{quote}{text}{quote}")
        except (SyntaxError, ValueError):
            return None
        return decoded if isinstance(decoded, str) else None
    return None


def _blob_text(
    value: object,
    container: str,
    question_marker: str,
    answer_marker: str,
) -> str:
    raw = str(value)
    if container == "python_list":
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = None
        marker = re.compile(
            rf"_*\s*(?:{re.escape(question_marker)}|{re.escape(answer_marker)})\s*:"
        )
        if isinstance(parsed, (list, tuple)):
            if parsed and all(isinstance(item, str) and marker.match(item) for item in parsed):
                return "".join(parsed)
            return raw
        wrapper = re.fullmatch(r"\s*\[\s*(['\"])(.*)(['\"])\s*\]\s*", raw, re.DOTALL)
        if wrapper is None:
            return raw
        body = wrapper.group(2)
        separator = re.compile(rf"['\"]\s*,\s*['\"](?={marker.pattern})")
        parts = separator.split(body)
        if any(
            marker.match(part) is None or re.search(r"['\"]\s*,\s*['\"]", part)
            for part in parts
        ):
            return raw
        decoded_parts = []
        for part in parts:
            decoded = _decode_python_item(part)
            if decoded is None:
                return raw
            decoded_parts.append(decoded)
        return "".join(decoded_parts)
    return raw


def _split_blob(
    text: str,
    question_marker: str,
    answer_marker: str,
    *,
    preserve_payload: bool = False,
) -> list[tuple[str, str | None]]:
    pattern = re.compile(rf"_*\s*({re.escape(question_marker)}|{re.escape(answer_marker)})\s*:")
    matches = list(pattern.finditer(text))
    if not matches or text[: matches[0].start()].strip(" _\n\t"):
        return []
    pairs: list[tuple[str, str | None]] = []
    expected = question_marker
    question: str | None = None
    for index, match in enumerate(matches):
        marker = match.group(1)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_chunk = text[match.end() : end]
        if preserve_payload:
            chunk = raw_chunk[1:] if raw_chunk.startswith(" ") else raw_chunk
        else:
            chunk = raw_chunk.strip().strip("_").strip()
        if marker != expected:
            return []
        if marker == question_marker:
            if not chunk.strip():
                return []
            question = chunk
            expected = answer_marker
        else:
            if question is None:
                return []
            pairs.append((question, chunk if chunk.strip() else None))
            question = None
            expected = question_marker
    return pairs if expected == question_marker else []


def _dialogue_literal(value: object) -> list[tuple[str, str, str | None]] | None:
    """Официальный dialogue: список троек query_id, вопрос, ответ."""
    if isinstance(value, (list, tuple)):
        parsed = value
    else:
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            return None
    if not isinstance(parsed, (list, tuple)) or not parsed:
        return None
    turns = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return None
        query_id, question, answer = item
        if (
            query_id is None
            or not str(query_id).strip()
            or question is None
            or not str(question).strip()
        ):
            return None
        turns.append(
            (
                str(query_id),
                str(question),
                None if answer is None else str(answer),
            )
        )
    return turns


def _unroll_blob(region: TableRegion, blob: dict[str, object]) -> GroupedTable:
    column = str(blob["column"])
    index = region.columns.index(column)
    rows: list[list[object]] = []
    groups: list[int] = []
    source_rows: list[int] = []
    query_ids: list[str] = []
    invalid = []
    for group, (source_row, row) in enumerate(zip(region.source_rows, region.rows)):
        value = row[index]
        container = str(blob["container"])
        question_marker = str(blob["question_marker"])
        answer_marker = str(blob["answer_marker"])
        literal = None if value is None else _dialogue_literal(value)
        if literal is not None:
            pairs = [(question, answer) for _query_id, question, answer in literal]
        elif value is None:
            pairs = []
        else:
            pairs = _split_blob(
                _blob_text(value, container, question_marker, answer_marker),
                question_marker,
                answer_marker,
                preserve_payload=container == "python_list",
            )
        if not pairs:
            invalid.append({"row": source_row, "text": str(value)[:120]})
            continue
        for turn, (question, answer) in enumerate(pairs, start=1):
            rows.append([question, answer, *row])
            groups.append(group)
            source_rows.append(source_row)
            query_ids.append(
                literal[turn - 1][0] if literal is not None else f"row-{source_row}-t{turn}"
            )
    if invalid:
        raise LayoutError(
            "Dialogue blob содержит непустые строки, которые нельзя полностью развернуть",
            invalid_rows=invalid[:10],
            invalid_count=len(invalid),
        )
    return GroupedTable(
        columns=["__blob_input_query", "__blob_output_answer", *region.columns],
        rows=rows,
        source_rows=source_rows,
        group_index=_number_groups(groups),
        query_id_override=query_ids,
        accounting={"source_rows": len(region.rows), "unrolled_pairs": len(rows)},
    )


def apply_grouping(sheet: RawSheet, region: TableRegion, config: dict[str, object]) -> GroupedTable:
    grouping = config["grouping"]
    kind = grouping["kind"]
    if kind == "none":
        return GroupedTable(list(region.columns), region.rows, region.source_rows, None)
    if kind == "merged_rows":
        rows, groups, filled = _merged_rows(sheet, region)
        return GroupedTable(
            list(region.columns), rows, region.source_rows, groups,
            accounting={"n_groups": len(set(groups)), "merged_cells_filled": filled},
        )
    if kind == "column":
        groups = _column_groups(region, str(grouping["column"]))
        return GroupedTable(
            list(region.columns), region.rows, region.source_rows, groups,
            accounting={"n_groups": len(set(groups))},
        )
    if kind == "blob_row":
        table = _unroll_blob(region, config["dialogue_blob"])
        table.accounting["n_groups"] = len(set(table.group_index or []))
        return table
    raise LayoutError(f"Неизвестная группировка: {kind}")
