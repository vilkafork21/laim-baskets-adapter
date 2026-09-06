"""Валидация контракта, единицы наблюдения и Decimal-агрегация."""

from __future__ import annotations

import ast
import json
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path

import pandas as pd

from . import formula as formula_lang

_VERSION = "laim-monitoring-metric.v2"
_LEGACY_VERSION = "laim-monitoring-metric.v1"
_UMR_VERSION = "laim-umr.v2"
_ASSESSMENT_MODES = {"qa", "turn_with_history", "dialogue"}
# Готовые способы score из v2: каждый — это просто формула (см. contract_formula).
# "formula" — метрика записана явно, как она определена в отчёте о валидации.
_METHOD_ROLES = {
    "identity": {"final_score": (1, 1)},
    "accuracy": {"prediction": (1, 1), "target": (1, 1)},
    "mean_criteria": {"criterion": (1, None)},
    "all_criteria": {"criterion": (1, None)},
    "majority": {"assessor_vote": (1, None)},
    "all_assessors": {"assessor_vote": (2, None)},
}
_ROLES = {"final_score", "criterion", "assessor_vote", "prediction", "target"}
_MISSING = {"fail", "exclude_unit", "exclude_value", "zero"}
_WEIGHT = "weight"


class MonitoringContractError(ValueError):
    """Контракт или данные не позволяют получить однозначный результат."""


def _require(mapping: dict, name: str, expected=None):
    if name not in mapping:
        raise MonitoringContractError(f"monitoring_metric не содержит {name}")
    value = mapping[name]
    if expected is not None and value not in expected:
        raise MonitoringContractError(f"Недопустимое {name}: {value!r}")
    return value


def _decimal(value: object, name: str = "value") -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise MonitoringContractError(f"{name} не является Decimal: {value!r}") from exc
    if not result.is_finite():
        raise MonitoringContractError(f"{name} должно быть конечным")
    return result


def _upgrade_contract(payload: dict) -> dict:
    version = payload.get("contract_version")
    if version == _VERSION:
        return deepcopy(payload)
    if version != _LEGACY_VERSION:
        raise MonitoringContractError(
            f"Неизвестная версия monitoring_metric: {version!r}"
        )
    if payload.get("status") == "computed" and payload.get("evaluation_unit") != "turn":
        raise MonitoringContractError(
            "monitoring_metric.v1 dialogue нельзя восстановить без turn_index"
        )
    upgraded = deepcopy(payload)
    upgraded["contract_version"] = _VERSION
    upgraded["umr_version"] = _UMR_VERSION
    upgraded["assessment_mode"] = "qa"
    upgraded["status"] = (
        "not_computable" if upgraded.get("status") == "not_evaluable" else upgraded.get("status")
    )
    upgraded.pop("evaluation_unit", None)
    upgraded.pop("group_column", None)
    return upgraded


def validate_monitoring_metric(payload: object, *, require_computed: bool = True) -> dict:
    if not isinstance(payload, dict):
        raise MonitoringContractError("monitoring_metric должен быть JSON object")
    contract = _upgrade_contract(payload)
    if _require(contract, "umr_version") != _UMR_VERSION:
        raise MonitoringContractError(f"Неизвестная версия UMR: {contract.get('umr_version')!r}")
    status = _require(contract, "status", {"computed", "not_computable"})
    if status != "computed":
        if require_computed:
            raise MonitoringContractError(
                f"monitoring_metric невычислим: {contract.get('reason', 'причина не указана')}"
            )
        return contract

    _require(contract, "basket_id")
    _require(contract, "name")
    if _require(contract, "score_column") != "main_metric":
        raise MonitoringContractError("Единственная каноническая score-колонка: main_metric")
    _require(contract, "assessment_mode", _ASSESSMENT_MODES)

    scoring = _require(contract, "scoring")
    if not isinstance(scoring, dict):
        raise MonitoringContractError("scoring должен быть object")
    method = _require(scoring, "method", set(_METHOD_ROLES) | {"formula"})
    sources = _require(scoring, "sources")
    if not isinstance(sources, list) or not sources:
        raise MonitoringContractError("scoring.sources должен быть непустым списком")
    source_ids = set()
    names = set()
    role_counts: dict[str, int] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise MonitoringContractError("Каждый scoring source должен быть object")
        source_id = _require(source, "source_id")
        if source_id in source_ids:
            raise MonitoringContractError(f"Повторяется source_id: {source_id}")
        source_ids.add(source_id)
        _require(source, "column_name")
        role = _require(source, "role", _ROLES)
        role_counts[role] = role_counts.get(role, 0) + 1
        # Адаптер публикует значения уже нормализованными: числа и метки.
        _require(source, "normalization", {"numeric", "label"})
        _require(source, "polarity", {"direct"})
        name = source_name(source)
        if not name.isidentifier() or name in formula_lang.HELPERS or name == _WEIGHT:
            raise MonitoringContractError(f"Недопустимое имя входа формулы: {name!r}")
        if name in names:
            raise MonitoringContractError(f"Повторяется имя входа формулы: {name!r}")
        names.add(name)
    if method == "formula":
        if not isinstance(contract.get("formula"), str):
            raise MonitoringContractError("method=formula требует текст formula")
    else:
        expected_roles = _METHOD_ROLES[method]
        if set(role_counts) != set(expected_roles):
            raise MonitoringContractError(
                f"Метод {method} требует роли {sorted(expected_roles)}, получено {sorted(role_counts)}"
            )
        for role, (minimum, maximum) in expected_roles.items():
            count = role_counts[role]
            if count < minimum or (maximum is not None and count > maximum):
                raise MonitoringContractError(f"Недопустимое число источников роли {role}: {count}")
    missing_policy = _require(scoring, "missing_policy", _MISSING)
    denominator = scoring.get("majority_denominator")
    if method == "majority" and denominator not in {"declared", "present"}:
        raise MonitoringContractError("majority требует denominator declared или present")
    if method != "majority" and denominator is not None:
        raise MonitoringContractError("majority_denominator допустим только для majority")
    aggregation = _require(contract, "aggregation")
    reducer = _require(aggregation, "method", {"mean", "frequency_weighted_mean"})
    weight_column = aggregation.get("weight_column")
    if reducer == "frequency_weighted_mean" and weight_column != "input_query_count":
        raise MonitoringContractError("Weighted mean требует input_query_count")
    if reducer == "mean" and weight_column is not None:
        raise MonitoringContractError("mean не должен объявлять weight_column")

    try:
        parsed = formula_lang.parse(contract_formula(contract))
    except formula_lang.FormulaError as exc:
        raise MonitoringContractError(f"Формула КМ: {exc}") from exc
    unknown = [name for name in parsed.inputs if name not in names and name != _WEIGHT]
    if unknown:
        raise MonitoringContractError(
            f"Формула ссылается на входы {unknown}, объявлены {sorted(names)} и {_WEIGHT}"
        )
    baseline = _require(contract, "baseline")
    _decimal(_require(baseline, "value"), "baseline.value")
    _decimal(_require(baseline, "recomputed_value"), "baseline.recomputed_value")
    _require(baseline, "scale", {"ratio", "raw"})
    validation = _require(contract, "primary_validation")
    if validation.get("affects_monitoring") is not False:
        raise MonitoringContractError("Primary validation threshold не должен влиять на monitoring")
    return contract


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _key(value: object) -> tuple[str, str]:
    if isinstance(value, bool):
        return "bool", str(value)
    if isinstance(value, (int, float, Decimal)) and not _blank(value):
        return "number", str(_decimal(value).normalize())
    return "text", str(value).strip().casefold()


def _constant(values: list[object], name: str) -> object | None:
    present = [value for value in values if not _blank(value)]
    if not present:
        return None
    if len({_key(value) for value in present}) != 1:
        raise MonitoringContractError(f"{name} не константен внутри dialogue")
    return present[0]


def _source_values(row, sources: list[dict]) -> dict[str, object]:
    result = {}
    for source in sources:
        column = source["column_name"]
        if column in row.index:
            result[source["source_id"]] = row[column]
    return result


def _turn_order(value: object) -> int | None:
    """Привести turn_index из любого транспорта (int/float/str) к int >= 1."""
    if _blank(value) or isinstance(value, bool):
        return None
    try:
        decimal_index = _decimal(value)
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
        key = ("solo", position) if _blank(group) else _key(group)
        label = f"row-{position}" if _blank(group) else group
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
        "input_query": "" if _blank(row["input_query"]) else str(row["input_query"]),
        "output_answer": "" if _blank(row["output_answer"]) else str(row["output_answer"]),
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
        result = fallback if _blank(value) else value
        key = _key(result)
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
            if "input_query_count" in row.index and not _blank(row["input_query_count"])
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
                "query_id": f"{group}-t{turn_index}" if _blank(query_id) else query_id,
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

    mode = _require(contract, "assessment_mode", _ASSESSMENT_MODES)
    flat_columns = ("query_id", "input_query", "output_answer")
    if all(name in frame for name in flat_columns):
        if "dialogue" in frame:
            raise MonitoringContractError(
                "UMR смешан: плоские колонки и dialogue одновременно; "
                "транспорт обязан быть либо flat, либо packed"
            )
        result = frame.copy()
        if any(_blank(value) for value in result["query_id"]):
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
        if any(_blank(value) for value in result["session_id"]):
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


def _load_tdc_monitoring(
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
        "input_query": "" if _blank(row["input_query"]) else str(row["input_query"]),
        "output_answer": "" if _blank(row["output_answer"]) else str(row["output_answer"]),
    }


def _turn_record(
    frame,
    position,
    sources,
    score_column,
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
        **_source_values(row, sources),
    }
    current = (
        _turn(row, len(history or ()) + 1)
        if mode == "turn_with_history" else _current_turn(row)
    )
    record["assessment_context"] = {"mode": mode, "current_turn": current}
    if mode == "turn_with_history":
        record["assessment_context"]["history"] = list(history or ())
    if score_column in frame:
        record[score_column] = row[score_column]
    return record


def _unitize(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    frame = normalize_umr(frame, contract)
    query_label = _unique_labels()
    frame["query_id"] = [
        query_label(value, f"row-{position}")
        for position, value in enumerate(frame["query_id"].tolist())
    ]
    sources = contract.get("scoring", {}).get("sources", [])
    score_column = contract.get("score_column", "main_metric")
    weighted = (
        contract.get("aggregation", {}).get("method")
        == "frequency_weighted_mean"
    )
    mode = _require(contract, "assessment_mode", _ASSESSMENT_MODES)

    records = []
    if mode == "qa":
        for position in range(len(frame)):
            records.append(_turn_record(
                frame, position, sources, score_column,
                mode=mode,
                group=None,
            ))
        return pd.DataFrame(records)

    groups = _ordered_groups(frame)
    if mode == "turn_with_history":
        for group, positions in groups:
            history = []
            for position in positions:
                records.append(_turn_record(
                    frame, position, sources, score_column,
                    mode=mode, group=group, history=history,
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
        for source in sources:
            column = source["column_name"]
            if column in part:
                record[source["source_id"]] = _constant(
                    part[column].tolist(), column
                )
        if score_column in part:
            record[score_column] = _constant(part[score_column].tolist(), score_column)
        records.append(record)
    return pd.DataFrame(records)


def unitize(frame: pd.DataFrame, payload: dict) -> pd.DataFrame:
    return _unitize(frame, validate_monitoring_metric(payload))


def source_name(source: dict) -> str:
    """Имя входа в формуле: явный name либо source_id."""
    return str(source.get("name") or source["source_id"])


def contract_formula(contract: dict) -> str:
    """Текст формулы КМ: явный `formula`, иначе синтез из готового метода v2."""
    explicit = contract.get("formula")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    scoring = contract["scoring"]
    method = scoring["method"]
    if method == "formula":
        raise MonitoringContractError("method=formula требует текст formula")
    by_role: dict[str, list[str]] = {}
    for source in scoring["sources"]:
        by_role.setdefault(source["role"], []).append(source_name(source))
    policy = scoring.get("missing_policy", "fail")

    def rows(names: list[str]) -> list[str]:
        return [f"fillna({name}, 0)" if policy == "zero" else name for name in names]

    if method == "identity":
        unit = rows(by_role["final_score"])[0]
    elif method == "accuracy":
        unit = f"{by_role['prediction'][0]} == {by_role['target'][0]}"
    elif method == "mean_criteria":
        names = rows(by_role["criterion"])
        unit = (
            f"avg({', '.join(names)})" if policy == "exclude_value" and len(names) > 1
            else f"({' + '.join(names)}) / {len(names)}" if len(names) > 1
            else names[0]
        )
    elif method in ("all_criteria", "all_assessors"):
        names = rows(by_role["criterion" if method == "all_criteria" else "assessor_vote"])
        unit = f"min({', '.join(names)})" if len(names) > 1 else names[0]
    else:  # majority
        declared = scoring.get("majority_denominator") == "declared"
        unit = f"majority({', '.join(by_role['assessor_vote'])}{', declared=True' if declared else ''})"
    weighted = contract.get("aggregation", {}).get("method") == "frequency_weighted_mean"
    return f"wmean({unit}, {_WEIGHT})" if weighted else f"mean({unit})"


def _normalized_column(units: pd.DataFrame, source: dict) -> pd.Series:
    """Метки — текст, числа — float; пустое — NaN."""
    raw = units[source["source_id"]]
    if source["normalization"] == "label":
        return raw.map(lambda v: None if _blank(v) else str(v)).astype(object)
    return pd.to_numeric(raw, errors="coerce").astype("float64")


def formula_columns(units: pd.DataFrame, contract: dict) -> dict[str, pd.Series]:
    """Входы формулы, доступные в units; отсутствующие колонки просто не попадают."""
    columns = {
        source_name(source): _normalized_column(units, source)
        for source in contract["scoring"]["sources"]
        if source["source_id"] in units
    }
    if "input_query_count" in units:
        columns[_WEIGHT] = pd.to_numeric(units["input_query_count"], errors="coerce").astype("float64")
    return columns


def _check_blanks(columns: dict[str, pd.Series], names: tuple[str, ...], policy: str) -> None:
    if policy != "fail":
        return
    for name in names:
        if name in columns and columns[name].isna().any():
            raise MonitoringContractError(f"missing_policy=fail: вход {name!r} содержит пропуски")


def unit_scores(units: pd.DataFrame, contract: dict) -> pd.Series:
    """Построчный score единиц: построчная часть формулы mean(E)/wmean(E, w).

    Для формул без построчной части (macro-F1 и т.п.) — совпадение prediction
    с target, если обе роли есть, иначе NaN: такой метрике построчный score
    не нужен, он служит только дрифт-тестам и примерам в отчёте.
    """
    parsed = formula_lang.parse(contract_formula(contract))
    columns = formula_columns(units, contract)
    unit = parsed.unit_expression()
    if unit is None:
        by_role = {source["role"]: source_name(source) for source in contract["scoring"]["sources"]}
        if "prediction" in by_role and "target" in by_role:
            unit = formula_lang.parse(f"{by_role['prediction']} == {by_role['target']}")
        else:
            return pd.Series(float("nan"), index=units.index, dtype="float64")
    missing = [name for name in unit.inputs if name not in columns]
    if missing:
        raise MonitoringContractError(f"Нет входов формулы в единицах оценки: {missing}")
    _check_blanks(columns, unit.inputs, contract["scoring"]["missing_policy"])
    try:
        return unit.evaluate_rows(columns).astype("float64")
    except formula_lang.FormulaError as exc:
        raise MonitoringContractError(f"Формула КМ: {exc}") from exc


def score_units(units: pd.DataFrame, payload: dict) -> pd.Series:
    return unit_scores(units, validate_monitoring_metric(payload))


def aggregate_main_metric(frame: pd.DataFrame, payload: dict) -> dict[str, object]:
    """КМ по единицам оценки той же формулой, что baseline на корзине.

    Входы формулы — колонки контракта (разметка судьи + ответ агента). Если их
    нет, а есть только построчный main_metric (старый выход ассесора), берётся
    его среднее; формула в результате показывает, что именно посчитано.
    """
    contract = validate_monitoring_metric(payload)
    units = unitize(frame, contract)
    policy = contract["scoring"]["missing_policy"]
    weighted = contract["aggregation"]["method"] == "frequency_weighted_mean"
    parsed = formula_lang.parse(contract_formula(contract))
    columns = formula_columns(units, contract)
    if any(name not in columns for name in parsed.inputs):
        if "main_metric" not in units:
            raise MonitoringContractError(
                f"Нет входов формулы {list(parsed.inputs)} и нет main_metric для агрегации"
            )
        scores = pd.to_numeric(units["main_metric"], errors="coerce").astype("float64")
        columns["main_metric"] = scores.fillna(0.0) if policy == "zero" else scores
        parsed = formula_lang.parse(f"wmean(main_metric, {_WEIGHT})" if weighted else "mean(main_metric)")
    _check_blanks(columns, parsed.inputs, policy)
    try:
        value = parsed.evaluate(columns)
        unit = parsed.unit_expression()
        scored = (
            unit.evaluate_rows(columns).notna() if unit is not None
            else pd.concat([columns[name] for name in parsed.inputs], axis=1).notna().all(axis=1)
        )
    except formula_lang.FormulaError as exc:
        raise MonitoringContractError(f"Формула КМ: {exc}") from exc
    if pd.isna(value) or not scored.any():
        raise MonitoringContractError("Нет оцененных единиц")
    weights = columns[_WEIGHT] if weighted else pd.Series(1.0, index=units.index)
    return {
        "name": contract["name"],
        "value": float(value),
        "formula": parsed.text,
        "total_units": int(len(units)),
        "scored_units": int(scored.sum()),
        "excluded_units": int((~scored).sum()),
        "weight_sum": float(weights[scored].sum()),
    }


def broadcast_scores(frame: pd.DataFrame, units: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    if len(units) != len(scores):
        raise MonitoringContractError("Число scores не совпадает с числом units")
    result = frame.copy()
    result["main_metric"] = None
    result["assessment_unit_id"] = None
    score_column = result.columns.get_loc("main_metric")
    unit_column = result.columns.get_loc("assessment_unit_id")
    for unit_id, positions, score in zip(
        units["_unit_id"], units["_row_positions"], scores.tolist()
    ):
        for position in positions:
            result.iat[position, score_column] = score
            result.iat[position, unit_column] = unit_id
    result["main_metric"] = pd.to_numeric(result["main_metric"], errors="coerce")
    return result


def _drift_frame(frame: pd.DataFrame, contract: dict, *, require_target: bool) -> pd.DataFrame:
    units = _unitize(frame, contract)
    if require_target and "main_metric" not in units:
        raise MonitoringContractError("Для drift отсутствует main_metric")
    target = pd.to_numeric(
        units["main_metric"] if "main_metric" in units else pd.Series([None] * len(units)),
        errors="coerce",
    )
    missing_policy = contract.get("scoring", {}).get("missing_policy", "fail")
    if require_target and missing_policy == "fail" and target.isna().any():
        raise MonitoringContractError("Для drift обнаружен пустой main_metric")
    mode = contract["assessment_mode"]
    if mode == "dialogue":
        questions = [
            json.dumps(
                [turn["input_query"] for turn in dialogue],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for dialogue in units["dialogue"]
        ]
    elif mode == "turn_with_history":
        questions = [
            json.dumps(
                [
                    *(turn["input_query"] for turn in context["history"]),
                    context["current_turn"]["input_query"],
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for context in units["assessment_context"]
        ]
    else:
        questions = units["input_query"].astype(str).tolist()
    group_ids = [
        unit_id if _blank(group_id) else group_id
        for group_id, unit_id in zip(units["_group_id"], units["_unit_id"])
    ]
    result = pd.DataFrame({
        "question": questions,
        "answer": [""] * len(units),
        "target": target,
        "reference_group_id": group_ids,
    })
    return result.dropna(subset=["target"]).reset_index(drop=True) if require_target else result


def prepare_drift_frames(
    reference_umr: pd.DataFrame,
    monitoring_umr: pd.DataFrame | bytes | bytearray | str | Path,
    payload: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contract = validate_monitoring_metric(payload, require_computed=False)
    if "assessment_mode" not in contract:
        # Контракт без MeasurementPlan (plan-less not_computable) не несёт
        # режим оценки — drift честно отказывается вместо невнятного падения
        # внутри нормализации.
        raise MonitoringContractError(
            "Drift не вычисляется: monitoring_metric не содержит assessment_mode "
            f"(status={contract.get('status')!r}, reason={contract.get('reason')!r})"
        )
    # normalize_umr выполнит _unitize внутри _drift_frame — второй проход не нужен
    monitoring_umr = _load_tdc_monitoring(monitoring_umr)
    return (
        _drift_frame(reference_umr, contract, require_target=True),
        _drift_frame(monitoring_umr, contract, require_target=False),
    )
