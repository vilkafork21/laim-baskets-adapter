"""Физика плана метрики: контракт источников, форма данных, шкалы.

Проверяется исполнимость плана движком и согласие с формой корзины —
без span-доказательств: цитаты модели идут в журнал, не в гейты.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from decimal import Decimal, InvalidOperation

import jsonschema
from openpyxl.utils import column_index_from_string

from ..errors import MeasurementPlanError, NotEvaluableError
from ..llm.schemas import METRIC_SCHEMA
from ..models import MeasurementPlan, ResolvedLayout
from ..reading.formulas import is_row_local_formula
from ..reading.xlsx_reader import RawSheet
from ..transform.values import blank as _blank
from ..transform.values import normalize_key

logger = logging.getLogger(__name__)

_CELL_REF = re.compile(r"\$?([A-Z]{1,3})\$?[0-9]+")


def decimal_value(value: object) -> Decimal:
    text = re.sub(r"[\s   ]", "", str(value)).replace(",", ".")
    try:
        result = Decimal(text)
    except InvalidOperation as exc:
        raise MeasurementPlanError("Значение не является Decimal", value=str(value)) from exc
    if not result.is_finite():
        raise MeasurementPlanError("Значение должно быть конечным", value=str(value))
    return result


def reported_quantum(plan: MeasurementPlan) -> Decimal:
    """Один последний опубликованный разряд в канонической шкале плана."""
    if plan.reported_value is None:
        return Decimal(1)
    return Decimal(1).scaleb(plan.reported_value.as_tuple().exponent)


def _validate_source_contract(method: str, sources: list[dict[str, object]]) -> None:
    counts = Counter(source["role"] for source in sources)
    expected = {
        "identity": counts == Counter({"final_score": 1}),
        "accuracy": counts == Counter({"prediction": 1, "target": 1}),
        "mean_criteria": counts["criterion"] >= 2 and len(counts) == 1,
        "all_criteria": counts["criterion"] >= 2 and len(counts) == 1,
        "majority": counts["assessor_vote"] >= 2 and len(counts) == 1,
        "all_assessors": counts["assessor_vote"] >= 2 and len(counts) == 1,
    }
    if not expected[method]:
        raise NotEvaluableError(
            "Метод КМ несовместим с ролями источников", method=method, roles=dict(counts)
        )
    for source in sources:
        normalization = source["normalization"]
        if source["role"] in ("prediction", "target") and normalization != "label":
            raise NotEvaluableError("Accuracy сравнивает labels, а не числовые class IDs")
        if source["role"] not in ("prediction", "target") and normalization == "label":
            raise NotEvaluableError("Числовой score не может иметь normalization=label")
        if normalization == "label" and source["polarity"] != "direct":
            raise NotEvaluableError("Label нельзя инвертировать арифметически")
        if isinstance(normalization, dict):
            normalized: dict[str, Decimal] = {}
            for key, value in normalization.items():
                norm = normalize_key(key)
                mapped = decimal_value(value)
                if norm in normalized:
                    raise NotEvaluableError("Коллизия ключей value_map после нормализации", key=key)
                normalized[norm] = mapped


def _scores_group_scoped(frame, layout: ResolvedLayout, column_ids: list[str]) -> bool:
    """Каждая группа имеет ровно один непустой score и есть группы из >1 строк."""
    groups: dict[object, list[int]] = {}
    for index, group in enumerate(frame["reference_group_id"].tolist()):
        groups.setdefault(group, []).append(index)
    if all(len(indexes) <= 1 for indexes in groups.values()):
        return False
    return all(
        sum(not _blank(frame[layout.column_names[column]].iloc[index]) for index in indexes) == 1
        for column in column_ids
        for indexes in groups.values()
    )


def vertically_merged_source(sheet: RawSheet, layout: ResolvedLayout, column_id: str) -> bool:
    column = column_index_from_string(column_id) - 1
    first = layout.first_data_row - 1
    last = layout.last_data_row - 1
    return any(
        row2 > row1 and column1 <= column <= column2 and row2 >= first and row1 <= last
        for row1, column1, row2, column2 in sheet.merged
    )


def _formula_components(
    sheet: RawSheet, layout: ResolvedLayout, formula_sources: dict[str, tuple[int, ...]]
) -> dict[str, list[str]]:
    """Колонки, на которые ссылается построчная формула score-колонки."""
    known = set(layout.region.column_letters)
    result: dict[str, list[str]] = {}
    for column_id, rows in formula_sources.items():
        row0 = rows[0] - 1
        column0 = layout.region.column_letters.index(column_id)
        formula = sheet.formulas.get((row0, column0), "")
        result[column_id] = sorted(
            {
                letter
                for letter in _CELL_REF.findall(formula)
                if letter in known and letter != column_id
            }
        )
    return result


def _assessment_mode(sheet: RawSheet, layout: ResolvedLayout, frame, column_ids: list[str]) -> str:
    """Режим оценки определяет физическая форма корзины, не мнение модели."""
    kind = layout.grouping["kind"]
    if kind == "blob_row":
        return "dialogue"
    if kind == "none":
        return "qa"
    if kind == "column":
        return "turn_with_history"
    # merged_rows: score, физически заданный один раз на многострочную группу
    # (vertical merge либо единственная непустая ячейка), делает единицу
    # оценки dialogue независимо от вида таблицы.
    merged_sources = [
        column for column in column_ids if vertically_merged_source(sheet, layout, column)
    ]
    if merged_sources and len(merged_sources) == len(column_ids):
        return "dialogue"
    grouped = all(name in frame for name in ("reference_group_id", "turn_index"))
    if grouped and _scores_group_scoped(frame, layout, column_ids):
        return "dialogue"
    return "turn_with_history"


def resolve_measurement_plan(
    proposal: dict, layout: ResolvedLayout, frame, sheet: RawSheet
) -> MeasurementPlan:
    try:
        jsonschema.validate(proposal, METRIC_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise MeasurementPlanError(
            f"План не соответствует схеме: {exc.message}",
            path=exc.json_path,
        ) from exc

    method = proposal["method"]
    sources = [dict(source) for source in proposal["sources"]]
    _validate_source_contract(method, sources)
    column_ids = [source["column_id"] for source in sources]
    if len(column_ids) != len(set(column_ids)):
        raise NotEvaluableError("Одна физическая колонка повторена в плане")
    missing_columns = [column for column in column_ids if column not in layout.column_names]
    if missing_columns:
        raise NotEvaluableError(
            "План ссылается на неизвестные колонки",
            columns=missing_columns,
            known_columns=sorted(layout.column_names),
        )
    missing_values = {
        column: tuple(values) for column, values in proposal.get("missing_values", {}).items()
    }
    if set(missing_values) - set(column_ids):
        raise NotEvaluableError("missing_values ссылается на колонку вне score-источников")
    for tokens in missing_values.values():
        for token in tokens:
            try:
                decimal_value(token.rstrip("%").strip())
            except MeasurementPlanError:
                continue
            raise NotEvaluableError(
                "Нельзя исключать числовые оценки через missing_values", token=token
            )

    def absent(column, value):
        return _blank(value) or normalize_key(value) in {
            normalize_key(token) for token in missing_values.get(column, ())
        }

    if layout.weight is not None and layout.weight["column_id"] in column_ids:
        raise NotEvaluableError(
            "Weight-колонка не может одновременно быть источником score",
            column_id=layout.weight["column_id"],
        )
    formula_sources = {
        column: layout.formula_rows[column]
        for column in column_ids
        if column in layout.formula_rows
    }
    unsafe_formulas = {
        column: tuple(
            row
            for row in rows
            if not is_row_local_formula(
                sheet.formulas.get((row - 1, column_index_from_string(column) - 1), ""), row
            )
        )
        for column, rows in formula_sources.items()
    }
    unsafe_formulas = {column: rows for column, rows in unsafe_formulas.items() if rows}
    if unsafe_formulas:
        raise NotEvaluableError(
            "Этап S. Ожидалось: score с формулой только своей строки. "
            "Получено: межстрочные, внешние или недоказуемые ссылки. "
            "Действие: выберите исходные оценки либо сохраните статические значения.",
            formula_sources=unsafe_formulas,
            formula_components=_formula_components(sheet, layout, unsafe_formulas),
        )

    for column in column_ids:
        column0 = column_index_from_string(column) - 1
        for row1, col1, row2, col2 in sheet.merged:
            if row1 == row2 or not col1 <= column0 <= col2:
                continue
            indexes = [
                i
                for i, row in enumerate(frame["source_row_id"].tolist())
                if row1 < int(row) <= row2 + 1
            ]
            units = {
                frame["reference_group_id"].iloc[i]
                if "reference_group_id" in frame
                else int(frame["source_row_id"].iloc[i])
                for i in indexes
            }
            if len(units) > 1:
                raise NotEvaluableError(
                    "Одна объединённая оценка пересекает несколько групп",
                    column_id=column,
                    first_row=row1 + 1,
                    last_row=row2 + 1,
                )

    assessment_mode = proposal.get("assessment_mode") or _assessment_mode(
        sheet, layout, frame, column_ids
    )
    grouped = all(name in frame for name in ("reference_group_id", "turn_index"))
    if assessment_mode in ("dialogue", "turn_with_history") and not grouped:
        raise NotEvaluableError("Для оценки с историей или dialogue нужны границы групп")
    if assessment_mode == "qa" and layout.grouping["kind"] != "none":
        raise NotEvaluableError("qa несовместим с группировкой: уточните единицу по методике")
    if layout.grouping["kind"] == "blob_row" and assessment_mode != "dialogue":
        raise NotEvaluableError("Оценка контейнера dialogue не размножается на отдельные turns")
    evaluation_unit = "dialogue" if assessment_mode == "dialogue" else "turn"
    if evaluation_unit == "turn" and layout.grouping["kind"] == "merged_rows":
        merged_sources = [
            column for column in column_ids if vertically_merged_source(sheet, layout, column)
        ]
        if merged_sources:
            raise NotEvaluableError(
                "Вертикально merged source нельзя считать по turn: это перевзвешивает dialogue",
                columns=merged_sources,
            )
    _validate_source_contract(method, sources)

    reducer = proposal["reducer"]
    weighted = reducer == "frequency_weighted_mean"
    if evaluation_unit == "dialogue":
        groups: dict[object, list[int]] = {}
        for index, group in enumerate(frame["reference_group_id"].tolist()):
            groups.setdefault(group, []).append(index)
        for column in column_ids:
            name = layout.column_names[column]
            if any(
                len(
                    {
                        normalize_key(frame[name].iloc[index])
                        for index in indexes
                        if not absent(column, frame[name].iloc[index])
                    }
                )
                > 1
                for indexes in groups.values()
            ):
                raise NotEvaluableError(
                    "Источник КМ меняется внутри dialogue и не является одним dialogue score",
                    column_id=column,
                )
        if weighted and any(
            len({decimal_value(frame["input_query_count"].iloc[index]) for index in indexes}) > 1
            for indexes in groups.values()
        ):
            raise NotEvaluableError("Weight не константен внутри dialogue")
        any_missing = any(
            all(absent(column, frame[layout.column_names[column]].iloc[index]) for index in indexes)
            for indexes in groups.values()
            for column in column_ids
        )
    else:
        any_missing = any(
            absent(column, value)
            for column in column_ids
            for value in frame[layout.column_names[column]].tolist()
        )
    missing_policy = proposal["missing_policy"]
    if any_missing and missing_policy == "fail":
        raise NotEvaluableError("В источниках КМ есть пропуски, а missing_policy=fail")

    denominator = proposal["majority_denominator"]
    if method == "majority":
        if any_missing and denominator is None:
            raise NotEvaluableError("При неполных голосах majority требует явный denominator")
        denominator = denominator or "declared"
    elif denominator is not None:
        raise MeasurementPlanError("majority_denominator допустим только для majority")

    if weighted:
        if layout.weight is None:
            raise NotEvaluableError("Взвешенная КМ требует физическую weight-колонку")
        weight_id = layout.weight["column_id"]
        if weight_id in layout.formula_rows:
            raise NotEvaluableError(
                "Кеш Excel-формулы нельзя использовать как frequency weight",
                rows=layout.formula_rows[weight_id],
            )

    logger.info(
        "MeasurementPlan: метрика %r, метод %s, reducer %s, источников %d, "
        "missing_policy %s, режим %s, шкала %s, порог %s",
        proposal["metric_name"],
        method,
        reducer,
        len(sources),
        missing_policy,
        assessment_mode,
        proposal["scale"],
        None,
    )
    return MeasurementPlan(
        basket_id=layout.basket_id,
        metric_name=proposal["metric_name"],
        assessment_mode=assessment_mode,
        method=method,
        sources=tuple(sources),
        missing_policy=missing_policy,
        majority_denominator=denominator,
        reducer=reducer,
        threshold=None,
        comparator=None,
        scale=proposal["scale"],
        precision=0,
        reported_value=None,
        reported_raw=None,
        evidence={key: tuple(values) for key, values in proposal["quotes"].items()},
        missing_values=missing_values,
    )
