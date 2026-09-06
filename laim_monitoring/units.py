"""Единицы оценки: канонизация UMR и сборка units по assessment_mode."""

from __future__ import annotations

import ast
import json
from io import BytesIO
from pathlib import Path

import pandas as pd

from .contract import ASSESSMENT_MODES, require, uses_weight
from .errors import MonitoringContractError
from .values import blank, to_decimal, value_key


def _constant(values: list[object], name: str) -> object | None:
    present = [value for value in values if not blank(value)]
    if not present:
        return None
    if len({value_key(value) for value in present}) != 1:
        raise MonitoringContractError(f"{name} не константен внутри dialogue")
    return present[0]


def _input_values(row, inputs: list[dict]) -> dict[str, object]:
    return {item["name"]: row[item["column"]] for item in inputs if item["column"] in row.index}


def _turn_order(value: object) -> int | None:
    """Привести turn_index из любого транспорта (int/float/str) к int >= 1."""
    if blank(value) or isinstance(value, bool):
        return None
    try:
        decimal_index = to_decimal(value)
    except MonitoringContractError:
        return None
    if decimal_index != decimal_index.to_integral_value() or decimal_index < 1:
        return None
    return int(decimal_index)


def _ordered_groups(frame: pd.DataFrame) -> list[tuple[object, list[int]]]:
    """Группы в порядке появления; внутри группы — по turn_index.

    Транспорт между портами меняет типы и теряет колонки, поэтому данные
    приводятся и достраиваются: непригодный или отсутствующий turn_index
    заменяется порядком строк, строка без группы становится своей группой.
    """
    group_values = (
        frame["reference_group_id"].tolist()
        if "reference_group_id" in frame else [None] * len(frame)
    )
    order_values = (
        frame["turn_index"].tolist()
        if "turn_index" in frame else [None] * len(frame)
    )
    groups: dict[object, tuple[object, list[tuple[int | None, int]]]] = {}
    for position, (group, raw_index) in enumerate(zip(group_values, order_values)):
        key = ("solo", position) if blank(group) else value_key(group)
        label = f"row-{position}" if blank(group) else group
        groups.setdefault(key, (label, []))[1].append((_turn_order(raw_index), position))
    result = []
    for label, indexed_positions in groups.values():
        indexes = [index for index, _position in indexed_positions]
        usable = (
            all(index is not None for index in indexes)
            and sorted(indexes) == list(range(1, len(indexes) + 1))
        )
        positions = (
            [position for _index, position in sorted(indexed_positions)]
            if usable else [position for _index, position in indexed_positions]
        )
        result.append((label, positions))
    return result


def _turn(row, order: int) -> dict[str, object]:
    return {
        "turn_index": order,
        "input_query": "" if blank(row["input_query"]) else str(row["input_query"]),
        "output_answer": "" if blank(row["output_answer"]) else str(row["output_answer"]),
    }


def _restore_array_delimiters(text: str) -> str:
    """str() многоходового ndarray разделяет внешние array(...) пробелом или
    переводом строки без запятой — вставляем запятые, не трогая содержимое
    строковых литералов (текст реплики может содержать ") array(")."""
    result = []
    quote = None
    escaped = False
    for position, char in enumerate(text):
        result.append(char)
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == ")":
            tail = position + 1
            while tail < len(text) and text[tail].isspace():
                tail += 1
            if text.startswith("array(", tail) and tail > position + 1:
                result.append(",")
    return "".join(result)


def _parse_sequence_text(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        pass
    for candidate in dict.fromkeys((text, _restore_array_delimiters(text))):
        try:
            return _numpy_literal(ast.parse(candidate, mode="eval"))
        except (SyntaxError, ValueError):
            continue
    return text


def _sequence(value: object, name: str) -> list[object]:
    if isinstance(value, str) and value.strip():
        value = _parse_sequence_text(value.strip())
    if not isinstance(value, (list, tuple)):
        tolist = getattr(value, "tolist", None)
        value = tolist() if callable(tolist) else value
    if not isinstance(value, (list, tuple)):
        raise MonitoringContractError(f"{name} должен быть последовательностью")
    return list(value)


def _numpy_literal(node: ast.AST) -> object:
    """Безопасно разбирает repr вложенных numpy array из parquet transport."""
    if isinstance(node, ast.Expression):
        return _numpy_literal(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_numpy_literal(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_numpy_literal(item) for item in node.elts)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "array"
        and len(node.args) == 1
        and all(
            keyword.arg == "dtype"
            and isinstance(keyword.value, (ast.Name, ast.Constant))
            for keyword in node.keywords
        )
    ):
        return _numpy_literal(node.args[0])
    raise ValueError("неподдерживаемый строковый literal")


def _unique_labels() -> "callable":
    """Счётчик меток: пустое значение получает fallback, повтор — суффикс #N."""
    seen: dict[tuple[str, str], int] = {}

    def label(value: object, fallback: str) -> object:
        result = fallback if blank(value) else value
        key = value_key(result)
        seen[key] = seen.get(key, 0) + 1
        return result if seen[key] == 1 else f"{result}#{seen[key]}"

    return label


def _unpack_dialogue_tdc(frame: pd.DataFrame) -> pd.DataFrame:
    """Развернуть packed dialogue в плоские строки с группами и порядком."""
    extra_columns = [
        name for name in frame.columns
        if name not in {"dialogue", "input_query_count", "reference_group_id", "turn_index"}
    ]
    group_label = _unique_labels()
    records = []
    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        group = group_label(
            row["session_id"] if "session_id" in row.index else None,
            f"session-{row_number}",
        )
        turns = _sequence(row["dialogue"], f"dialogue в строке {row_number}")
        if not turns:
            raise MonitoringContractError(
                f"dialogue в строке {row_number} не содержит turns"
            )
        common = {name: row[name] for name in extra_columns}
        input_query_count = (
            row["input_query_count"]
            if "input_query_count" in row.index and not blank(row["input_query_count"])
            else 1
        )
        for turn_index, turn in enumerate(turns, start=1):
            values = _sequence(
                turn,
                f"turn {turn_index} dialogue в строке {row_number}",
            )
            if len(values) != 3:
                raise MonitoringContractError(
                    f"turn {turn_index} dialogue в строке {row_number} должен содержать "
                    "query_id, input_query, output_answer"
                )
            query_id, input_query, output_answer = values
            record = dict(common)
            record.update({
                "query_id": f"{group}-t{turn_index}" if blank(query_id) else query_id,
                "input_query": input_query,
                "output_answer": output_answer,
                "reference_group_id": group,
                "turn_index": turn_index,
                "input_query_count": input_query_count,
            })
            records.append(record)
    return pd.DataFrame(records)


def normalize_umr(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Канонизировать UMR в формате тестового датасета (reference из корзины или
    monitoring из TDC) к плоской форме с reference_group_id/turn_index: строгость —
    к контракту метрики, к данным — коэрсия и достройка (транспорт между портами
    меняет типы и теряет колонки)."""
    if not isinstance(frame, pd.DataFrame):
        raise MonitoringContractError("UMR должен быть pandas.DataFrame")
    if frame.empty:
        raise MonitoringContractError("UMR пуст")

    mode = require(contract, "assessment_mode", ASSESSMENT_MODES)
    flat_columns = ("query_id", "input_query", "output_answer")
    if all(name in frame for name in flat_columns):
        if "dialogue" in frame:
            raise MonitoringContractError(
                "UMR смешан: плоские колонки и dialogue одновременно; "
                "транспорт обязан быть либо flat, либо packed"
            )
        result = frame.copy()
        if any(blank(value) for value in result["query_id"]):
            raise MonitoringContractError(
                "Плоский UMR содержит пустой query_id: NA/NaT/пустые "
                "идентификаторы не материализуются текстом"
            )
    elif "dialogue" in frame:
        result = _unpack_dialogue_tdc(frame)
    else:
        missing = [name for name in flat_columns if name not in frame]
        raise MonitoringContractError(
            f"UMR не соответствует ни packed dialogue, ни плоской форме: "
            f"плоская форма неполна, нет колонок {missing}"
        )
    if mode == "qa":
        return result

    if "reference_group_id" not in result:
        # Контекстные режимы без явного порядка обязаны нести session_id:
        # порядок строк сам по себе хронологию не доказывает.
        if "session_id" not in result:
            raise MonitoringContractError(
                "Контекстный плоский UMR требует session_id либо явные "
                "reference_group_id + turn_index"
            )
        if any(blank(value) for value in result["session_id"]):
            raise MonitoringContractError(
                "Контекстный плоский UMR содержит пустой session_id"
            )
        result["reference_group_id"] = result["session_id"]
    group_column: list[object] = [None] * len(result)
    order_column: list[int] = [1] * len(result)
    for group, positions in _ordered_groups(result):
        for order, position in enumerate(positions, start=1):
            group_column[position] = group
            order_column[position] = order
    result["reference_group_id"] = group_column
    result["turn_index"] = order_column
    if "input_query_count" not in result:
        result["input_query_count"] = 1
    return result


def load_monitoring_frame(
    value: pd.DataFrame | bytes | bytearray | str | Path,
) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, (bytes, bytearray)):
        source = BytesIO(value)
    elif isinstance(value, (str, Path)):
        source = Path(value)
        if not source.is_file():
            raise MonitoringContractError(
                f"Файл Monitoring-входа TDC не найден: {source}"
            )
    else:
        raise MonitoringContractError(
            "Monitoring-вход TDC должен быть pandas.DataFrame, parquet bytes "
            "или путём к parquet"
        )
    try:
        return pd.read_parquet(source)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise MonitoringContractError(
            "Monitoring-вход TDC не содержит читаемый parquet"
        ) from exc


def _current_turn(row) -> dict[str, object]:
    return {
        "input_query": "" if blank(row["input_query"]) else str(row["input_query"]),
        "output_answer": "" if blank(row["output_answer"]) else str(row["output_answer"]),
    }


def _turn_record(
    frame,
    position,
    inputs,
    *,
    mode,
    group=None,
    history=None,
):
    row = frame.iloc[position]
    record = {
        "_unit_id": row["query_id"],
        "_row_positions": (position,),
        "_group_id": group,
        "input_query": row["input_query"],
        "output_answer": row["output_answer"],
        "input_query_count": row.get("input_query_count", 1),
        **_input_values(row, inputs),
    }
    current = (
        _turn(row, len(history or ()) + 1)
        if mode == "turn_with_history" else _current_turn(row)
    )
    record["assessment_context"] = {"mode": mode, "current_turn": current}
    if mode == "turn_with_history":
        record["assessment_context"]["history"] = list(history or ())
    if "main_metric" in frame:
        record["main_metric"] = row["main_metric"]
    return record


def _unitize(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    frame = normalize_umr(frame, contract)
    query_label = _unique_labels()
    frame["query_id"] = [
        query_label(value, f"row-{position}")
        for position, value in enumerate(frame["query_id"].tolist())
    ]
    inputs = contract["inputs"]
    weighted = uses_weight(contract)
    mode = require(contract, "assessment_mode", ASSESSMENT_MODES)

    records = []
    if mode == "qa":
        for position in range(len(frame)):
            records.append(_turn_record(frame, position, inputs, mode=mode, group=None))
        return pd.DataFrame(records)

    groups = _ordered_groups(frame)
    if mode == "turn_with_history":
        for group, positions in groups:
            history = []
            for position in positions:
                records.append(_turn_record(
                    frame, position, inputs, mode=mode, group=group, history=history,
                ))
                history.append(_turn(frame.iloc[position], len(history) + 1))
        return pd.DataFrame(records)

    for group, positions in groups:
        part = frame.iloc[positions]
        turns = [
            _turn(frame.iloc[position], order)
            for order, position in enumerate(positions, start=1)
        ]
        record = {
            "_unit_id": group,
            "_row_positions": tuple(positions),
            "_group_id": group,
            "dialogue": turns,
            "assessment_context": {"mode": mode, "turns": turns},
            "input_query_count": _constant(
                part["input_query_count"].tolist(), "input_query_count"
            ) if weighted else 1,
        }
        for item in inputs:
            if item["column"] in part:
                record[item["name"]] = _constant(part[item["column"]].tolist(), item["column"])
        if "main_metric" in part:
            record["main_metric"] = _constant(part["main_metric"].tolist(), "main_metric")
        records.append(record)
    return pd.DataFrame(records)


def unitize(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Единицы оценки по уже проверенному контракту (см. validate_monitoring_metric)."""
    return _unitize(frame, contract)
