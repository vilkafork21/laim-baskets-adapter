"""Разрешение предложения LLM в проверенный MeasurementPlan.

План описывает, как считать КМ по корзине: формула из отчёта о валидации,
входы формулы (колонки корзины), единица оценки и заявленное в отчёте
значение. Всё, что влияет на число, обязано быть подтверждено spans
документов; числовая правильность плана доказывается отдельно — пересчётом
(см. metric.engine и llm.generate._reconciliation_gate).
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

import jsonschema
from openpyxl.utils import column_index_from_string

from laim_monitoring import FormulaError, parse_formula

from .contracts import MEASUREMENT_SCHEMA
from .errors import MeasurementPlanError, NotEvaluableError
from .models import MeasurementPlan, ResolvedLayout, RunContext
from .transform.values import blank as _blank, normalize_key

_NUMBER = re.compile(r"-?[0-9]+(?:[\s   ][0-9]{3})*(?:[.,][0-9]+)?\s*%?")
_CELL_REF = re.compile(r"\$?([A-Z]{1,3})\$?[0-9]+")


def decimal_value(value: object) -> Decimal:
    text = re.sub(r"[\s   ]", "", str(value)).replace(",", ".")
    try:
        result = Decimal(text)
    except InvalidOperation as exc:
        raise MeasurementPlanError("Значение не является Decimal", value=str(value)) from exc
    if not result.is_finite():
        raise MeasurementPlanError("Значение должно быть конечным", value=str(value))
    return result


def reported_quantum(plan: MeasurementPlan) -> Decimal:
    """Один последний опубликованный разряд в канонической шкале плана."""
    raw = str(plan.reported_raw or plan.reported_value).strip()
    percent = raw.endswith("%")
    token = raw.rstrip("%").strip()
    try:
        value = decimal_value(token)
    except MeasurementPlanError:
        value = plan.reported_value
    quantum = Decimal(1).scaleb(value.as_tuple().exponent)
    if plan.scale == "percent" and not percent and abs(value) <= 1:
        return quantum * 100
    if plan.scale == "ratio" and percent:
        return quantum / 100
    return quantum


def _span_index(context: RunContext) -> dict[str, tuple[str, str]]:
    return {
        span["id"]: (document["id"], span["text"])
        for document in context.documents
        for span in document["spans"]
    }


def _numbers(text: str) -> list[tuple[Decimal, bool]]:
    result = []
    for match in _NUMBER.finditer(text):
        raw = match.group(0).strip()
        percent = raw.endswith("%")
        try:
            value = decimal_value(raw.rstrip("%").strip())
        except MeasurementPlanError:
            continue
        result.append((value, percent))
    return result


def _span_has_value(text: str, target: Decimal, scale: str) -> bool:
    for value, percent in _numbers(text):
        normalized = value / 100 if percent and scale == "ratio" else value
        if normalized == target:
            return True
        if scale == "percent" and not percent and value <= 1 and value * 100 == target:
            # Доля без знака % в percent-релизе: "0.9736" отвечает цели 97.36.
            return True
    return False


def _percent_domain_value(text: str, target: Decimal) -> Decimal | None:
    """Каноническое значение цели в домене percent-релиза по токену её span.

    Голый токен-доля (<= 1) означает долю и приводится к процентам; токен со
    знаком % или больше единицы уже записан в процентах.
    """
    for value, percent in _numbers(text):
        if value == target:
            return target * 100 if not percent and target <= 1 else target
        if not percent and value <= 1 and value * 100 == target:
            return target
    return None


def _formula_components(
    context: RunContext,
    layout: ResolvedLayout,
    formula_sources: dict[str, tuple[int, ...]],
) -> dict[str, list[str]]:
    """Колонки, на которые ссылается построчная формула score-колонки."""
    sheet = context.sheets[layout.sheet_name]
    known = set(layout.region.column_letters)
    result: dict[str, list[str]] = {}
    for column_id, rows in formula_sources.items():
        row0 = rows[0] - 1
        column0 = layout.region.column_letters.index(column_id)
        formula = sheet.formulas.get((row0, column0), "")
        components = sorted({
            letter for letter in _CELL_REF.findall(formula)
            if letter in known and letter != column_id
        })
        result[column_id] = components
    return result


def _scores_group_scoped(frame, layout: ResolvedLayout, column_ids: list[str]) -> bool:
    """Каждая группа имеет ровно один непустой score и есть группы из >1 строк."""
    groups: dict[object, list[int]] = {}
    for index, group in enumerate(frame["reference_group_id"].tolist()):
        groups.setdefault(group, []).append(index)
    if all(len(indexes) <= 1 for indexes in groups.values()):
        return False
    return all(
        sum(
            not _blank(frame[layout.column_names[column]].iloc[index])
            for index in indexes
        ) == 1
        for column in column_ids
        for indexes in groups.values()
    )


def _vertically_merged_source(
    context: RunContext,
    layout: ResolvedLayout,
    column_id: str,
) -> bool:
    column = column_index_from_string(column_id) - 1
    first = layout.first_data_row - 1
    last = layout.last_data_row - 1
    return any(
        row2 > row1
        and column1 <= column <= column2
        and row2 >= first
        and row1 <= last
        for row1, column1, row2, column2 in context.sheets[layout.sheet_name].merged
    )


def _resolve_inputs(proposal, layout: ResolvedLayout, context: RunContext) -> list[dict[str, object]]:
    inputs = [dict(item) for item in proposal["inputs"]]
    column_ids = [item["column_id"] for item in inputs]
    names = [item["name"] for item in inputs]
    if len(set(column_ids)) != len(column_ids):
        raise NotEvaluableError("Одна физическая колонка повторена во входах формулы")
    if len(set(names)) != len(names):
        raise MeasurementPlanError("Имена входов формулы повторяются", names=names)
    unknown = [column for column in column_ids if column not in layout.column_names]
    if unknown:
        raise NotEvaluableError("Входы формулы ссылаются на неизвестные колонки", columns=unknown)
    if layout.weight is not None and layout.weight["column_id"] in column_ids:
        raise NotEvaluableError(
            "Weight-колонка не может одновременно быть входом формулы",
            column_id=layout.weight["column_id"],
        )
    formula_backed = {
        column: layout.formula_rows[column] for column in column_ids if column in layout.formula_rows
    }
    if formula_backed:
        raise NotEvaluableError(
            "Кешированные результаты Excel-формул нельзя использовать как вход КМ",
            formula_sources=formula_backed,
            formula_components=_formula_components(context, layout, formula_backed),
            repair_hint=(
                "Формульная колонка — производная от сырых оценок. Возьми входами "
                "колонки-компоненты formula_components и запиши формулу над ними."
            ),
        )
    return inputs


def _resolve_formula(proposal, inputs: list[dict[str, object]], layout: ResolvedLayout) -> str:
    try:
        parsed = parse_formula(proposal["formula"])
    except FormulaError as exc:
        raise MeasurementPlanError(f"Формула не принята: {exc}", formula=proposal["formula"]) from exc
    names = {item["name"] for item in inputs}
    unknown = [name for name in parsed.inputs if name not in names and name != "weight"]
    if unknown:
        raise MeasurementPlanError(
            "Формула ссылается на входы, которых нет в inputs", unknown=unknown, declared=sorted(names),
        )
    if "weight" in parsed.inputs and layout.weight is None:
        raise NotEvaluableError("Формула использует weight, а в корзине нет weight-колонки")
    return parsed.text


def _resolve_assessment_mode(proposal, layout: ResolvedLayout, frame, column_ids: list[str], context) -> str:
    """Единица оценки — по документам, но физическая форма корзины сильнее выбора модели."""
    kind = layout.grouping["kind"]
    mode = proposal["assessment_mode"]
    if kind == "blob_row":
        mode = "dialogue"
    elif kind == "none":
        mode = "qa"
    elif kind == "column" and mode == "dialogue":
        mode = "turn_with_history"
    if (
        mode != "dialogue"
        and kind == "merged_rows"
        and all(name in frame for name in ("reference_group_id", "turn_index"))
        and _scores_group_scoped(frame, layout, column_ids)
    ):
        # Оценка физически задана один раз на многострочную группу — единица диалог.
        mode = "dialogue"
    if mode != "dialogue" and kind == "merged_rows":
        merged = [column for column in column_ids if _vertically_merged_source(context, layout, column)]
        if merged:
            raise NotEvaluableError(
                "Вертикально merged вход нельзя считать по репликам: это перевзвешивает диалог",
                columns=merged,
            )
    return mode


def _resolve_release(proposal, spans, evidence):
    release = proposal["release"]
    threshold_raw, comparator = release["threshold"], release["comparator"]
    if (threshold_raw is None) != (comparator is None):
        raise MeasurementPlanError("threshold и comparator задаются вместе или оба null")
    threshold = decimal_value(threshold_raw) if threshold_raw is not None else None
    if threshold is not None:
        if not evidence["release"]:
            raise MeasurementPlanError("Порог остался без evidence")
        matches = any(_span_has_value(spans[s][1], threshold, release["scale"]) for s in evidence["release"])
        alternative = {"ratio": "percent", "percent": "ratio"}.get(release["scale"])
        if not matches and alternative and any(
            _span_has_value(spans[s][1], threshold, alternative) for s in evidence["release"]
        ):
            release["scale"] = alternative
            matches = True
        if not matches:
            raise NotEvaluableError("Порог не найден в cited evidence", threshold=str(threshold))
    return threshold, comparator, release["scale"], release["precision"]


def _resolve_reported(proposal, spans, evidence, validation_document: str, scale: str):
    state, reported = proposal["reported_value_state"], proposal["reported_value"]
    if state == "ambiguous":
        raise NotEvaluableError(
            "Validation report содержит несколько итоговых значений КМ без указания, какое целевое"
        )
    if (state == "unambiguous") != (reported is not None):
        raise MeasurementPlanError("reported_value_state не согласован с reported_value")
    if reported is None:
        return None, None, None
    value = decimal_value(reported["value"])
    span_id = reported["span_id"]
    candidates = [
        s for s, (document, text) in spans.items()
        if document == validation_document and _span_has_value(text, value, scale)
    ]
    if span_id not in spans or spans[span_id][0] != validation_document:
        raise NotEvaluableError(
            "Итоговая КМ должна ссылаться на span validation report",
            reported_span=span_id, candidate_spans=candidates[:8],
        )
    if not _span_has_value(spans[span_id][1], value, scale):
        raise NotEvaluableError(
            "reported_value не соответствует числу в cited span", candidate_spans=candidates[:8],
        )
    if span_id not in evidence["reported_value"]:
        raise MeasurementPlanError("reported_value span должен входить в evidence.reported_value")
    if scale == "percent":
        value = _percent_domain_value(spans[span_id][1], value) or value
    return value, reported["raw"], span_id


def resolve_measurement_plan(proposal, context: RunContext, layout: ResolvedLayout, frame) -> MeasurementPlan:
    try:
        jsonschema.validate(proposal, MEASUREMENT_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise MeasurementPlanError(f"MeasurementPlan не соответствует схеме: {exc.message}", path=exc.json_path) from exc
    if proposal["basket_id"] != context.basket_id:
        raise MeasurementPlanError("basket_id MeasurementPlan не совпадает с пакетом")
    roles = proposal["document_roles"]
    if set(roles.values()) != {document["id"] for document in context.documents}:
        raise MeasurementPlanError("Три роли документов должны образовывать точное one-to-one соответствие")

    spans = _span_index(context)
    evidence = {key: tuple(value) for key, value in proposal["evidence"].items()}
    unknown = sorted({s for values in evidence.values() for s in values if s not in spans})
    if unknown:
        raise MeasurementPlanError("Evidence ссылается на неизвестные spans", spans=unknown)
    for field in ("metric", "formula", "assessment_mode"):
        if not evidence[field]:
            raise MeasurementPlanError("Значимое поле плана осталось без evidence", field=field)

    inputs = _resolve_inputs(proposal, layout, context)
    formula = _resolve_formula(proposal, inputs, layout)
    column_ids = [item["column_id"] for item in inputs]
    assessment_mode = _resolve_assessment_mode(proposal, layout, frame, column_ids, context)
    threshold, comparator, scale, precision = _resolve_release(proposal, spans, evidence)
    reported_value, reported_raw, reported_span = _resolve_reported(
        proposal, spans, evidence, roles["validation_report"], scale,
    )
    return MeasurementPlan(
        basket_id=context.basket_id,
        metric_name=proposal["metric_name"],
        document_roles=dict(roles),
        assessment_mode=assessment_mode,
        formula=formula,
        inputs=tuple(inputs),
        threshold=threshold,
        comparator=comparator,
        scale=scale,
        precision=precision,
        reported_value=reported_value,
        reported_raw=reported_raw,
        reported_span_id=reported_span,
        evidence=evidence,
    )
