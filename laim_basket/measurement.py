"""Разрешение недоверенного предложения LLM в закрытый MeasurementPlan."""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation

import jsonschema
from openpyxl.utils import column_index_from_string

from .contracts import MEASUREMENT_SCHEMA
from .errors import MeasurementPlanError, NotEvaluableError
from .models import MeasurementPlan, ResolvedLayout, RunContext
from .transform.values import blank as _blank, normalize_key

_NUMBER = re.compile(r"-?[0-9]+(?:[\s   ][0-9]{3})*(?:[.,][0-9]+)?\s*%?")
_MACRO_REDUCER = re.compile(r"\bmacro\b|\bмакро\w*", re.IGNORECASE)
_WEIGHTED_REDUCER = re.compile(r"\bweighted\b|\bвзвеш\w*", re.IGNORECASE)


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


def _reducer_variants(text: str) -> set[str]:
    variants = set()
    if _MACRO_REDUCER.search(text):
        variants.add("mean")
    if _WEIGHTED_REDUCER.search(text):
        variants.add("frequency_weighted_mean")
    return variants


def _reducer_candidate_spans(validation_spans: dict[str, str]) -> dict[str, str]:
    items = list(validation_spans.items())
    indexes = set()
    for index, (_span_id, text) in enumerate(items):
        if _reducer_variants(text):
            indexes.add(index)
            if index + 1 < len(items):
                indexes.add(index + 1)
    return dict(items[index] for index in sorted(indexes)[:12])


def _reported_reducer_candidates(
    validation_spans: dict[str, str],
    reducer: str,
    scale: str,
) -> list[dict[str, str]]:
    items = list(validation_spans.items())
    candidates = []
    for index, (label_span_id, label) in enumerate(items[:-1]):
        if _reducer_variants(label) != {reducer} or len(re.findall(r"\w+", label)) > 3:
            continue
        value_span_id, raw = items[index + 1]
        numbers = _numbers(raw)
        if len(numbers) != 1:
            continue
        value, percent = numbers[0]
        normalized = value / 100 if percent and scale == "ratio" else value
        candidates.append({
            "value": str(normalized),
            "raw": raw,
            "span_id": value_span_id,
            "label_span_id": label_span_id,
        })
    return candidates


def _validate_reducer_evidence(
    reducer: str,
    evidence: tuple[str, ...],
    spans: dict[str, tuple[str, str]],
    validation_document_id: str,
) -> None:
    validation_spans = {
        span_id: text
        for span_id, (document_id, text) in spans.items()
        if document_id == validation_document_id
    }
    available = {
        variant
        for text in validation_spans.values()
        for variant in _reducer_variants(text)
    }
    if available != {"mean", "frequency_weighted_mean"}:
        return

    designations = {
        variant: {
            span_id: text
            for span_id, text in validation_spans.items()
            if _reducer_variants(text) == {variant} and len(re.findall(r"\w+", text)) >= 3
        }
        for variant in available
    }
    required = [variant for variant, designation in designations.items() if designation]
    candidates = _reducer_candidate_spans(validation_spans)
    if len(required) == 1 and reducer != required[0]:
        raise NotEvaluableError(
            "Validation report содержит macro и weighted и явно задает другой reducer",
            reducer=reducer,
            required_reducer=required[0],
            designation_spans=designations[required[0]],
            candidate_spans=candidates,
        )

    cited = {
        span_id: validation_spans[span_id]
        for span_id in evidence
        if span_id in validation_spans
    }
    cited_variants = {
        variant
        for text in cited.values()
        for variant in _reducer_variants(text)
    }
    if cited_variants != {reducer}:
        raise NotEvaluableError(
            "Validation report содержит macro и weighted, но evidence.reducer не выбирает один вариант",
            reducer=reducer,
            candidate_spans=candidates,
        )
    explicit = [span_id for span_id in cited if span_id in designations[reducer]]
    if not explicit:
        raise NotEvaluableError(
            "Выбор между macro и weighted не подтвержден отдельным текстовым указанием validation report",
            reducer=reducer,
            candidate_spans=candidates,
        )


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
        raise NotEvaluableError("Метод КМ несовместим с ролями источников", method=method, roles=dict(counts))
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


_CELL_REF = re.compile(r"\$?([A-Z]{1,3})\$?[0-9]+")


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


def _canonicalize_score(score: dict[str, object]) -> tuple[str, list[dict[str, object]]]:
    method = score["method"]
    sources = [dict(source) for source in score["sources"]]
    expected_role = {
        "identity": "final_score",
        "mean_criteria": "criterion",
        "all_criteria": "criterion",
        "majority": "assessor_vote",
        "all_assessors": "assessor_vote",
    }.get(method)
    if expected_role is not None:
        matching = [source for source in sources if source["role"] == expected_role]
        minimum = 1 if method == "identity" else 2
        if len(matching) >= minimum:
            sources = matching
    elif method == "accuracy":
        predictions = [source for source in sources if source["role"] == "prediction"]
        targets = [source for source in sources if source["role"] == "target"]
        if len(predictions) == len(targets) == 1:
            sources = [predictions[0], targets[0]]
    if len(sources) == 1 and (
        sources[0]["role"] == "final_score"
        or method in ("mean_criteria", "all_criteria")
    ):
        method = "identity"
        sources[0]["role"] = "final_score"
    score["method"] = method
    score["sources"] = sources
    return method, sources


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


def resolve_measurement_plan(
    proposal: dict[str, object],
    context: RunContext,
    layout: ResolvedLayout,
    frame,
) -> MeasurementPlan:
    try:
        jsonschema.validate(proposal, MEASUREMENT_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise MeasurementPlanError(f"MeasurementPlan не соответствует схеме: {exc.message}", path=exc.json_path) from exc
    if proposal["basket_id"] != context.basket_id:
        raise MeasurementPlanError("basket_id MeasurementPlan не совпадает с пакетом")

    document_ids = {document["id"] for document in context.documents}
    roles = proposal["document_roles"]
    if set(roles.values()) != document_ids:
        raise MeasurementPlanError("Три роли документов должны образовывать точное one-to-one соответствие")
    spans = _span_index(context)
    evidence = {key: tuple(value) for key, value in proposal["evidence"].items()}
    unknown = sorted({span_id for values in evidence.values() for span_id in values if span_id not in spans})
    if unknown:
        raise MeasurementPlanError("Evidence ссылается на неизвестные spans", spans=unknown)
    for field in ("metric", "score", "assessment_mode", "reducer"):
        if not evidence[field]:
            raise MeasurementPlanError("Значимое поле плана осталось без evidence", field=field)

    score = proposal["score"]
    method, sources = _canonicalize_score(score)
    column_ids = [source["column_id"] for source in sources]
    if len(column_ids) != len(set(column_ids)):
        raise NotEvaluableError("Одна физическая колонка повторена в MeasurementPlan")
    missing_columns = [column for column in column_ids if column not in layout.column_names]
    if missing_columns:
        raise NotEvaluableError("MeasurementPlan ссылается на неизвестные колонки", columns=missing_columns)
    if layout.weight is not None and layout.weight["column_id"] in column_ids:
        raise NotEvaluableError(
            "Weight-колонка не может одновременно быть источником score",
            column_id=layout.weight["column_id"],
        )
    formula_sources = {column: layout.formula_rows[column] for column in column_ids if column in layout.formula_rows}
    if formula_sources:
        raise NotEvaluableError(
            "Кешированные результаты Excel-формул нельзя использовать как score",
            formula_sources=formula_sources,
            formula_components=_formula_components(context, layout, formula_sources),
            repair_hint=(
                "Формульная колонка — производная от сырых оценок. Построй план "
                "по колонкам-компонентам formula_components (например, голоса "
                "разметчиков как assessor_vote), а не по кешу формулы."
            ),
        )
    _validate_reducer_evidence(
        proposal["reducer"]["method"],
        evidence["reducer"],
        spans,
        roles["validation_report"],
    )
    grouping_kind = layout.grouping["kind"]
    assessment_mode = proposal["assessment_mode"]
    if grouping_kind == "blob_row":
        assessment_mode = "dialogue"
    elif grouping_kind == "none":
        assessment_mode = "qa"
    elif grouping_kind == "column" and assessment_mode == "dialogue":
        assessment_mode = "turn_with_history"
    if (
        assessment_mode != "dialogue"
        and grouping_kind == "merged_rows"
        and all(name in frame for name in ("reference_group_id", "turn_index"))
        and _scores_group_scoped(frame, layout, column_ids)
    ):
        # Score физически задан один раз на многострочную группу — данные
        # определяют единицу оценки как dialogue независимо от выбора LLM.
        assessment_mode = "dialogue"
    evaluation_unit = "dialogue" if assessment_mode == "dialogue" else "turn"
    if evaluation_unit == "turn" and layout.grouping["kind"] == "merged_rows":
        merged_sources = [
            column for column in column_ids
            if _vertically_merged_source(context, layout, column)
        ]
        if merged_sources:
            raise NotEvaluableError(
                "Вертикально merged source нельзя считать по turn: это перевзвешивает dialogue",
                columns=merged_sources,
            )
    _validate_source_contract(method, sources)

    reducer = proposal["reducer"]["method"]
    weighted = reducer == "frequency_weighted_mean"

    if evaluation_unit == "dialogue":
        group_values = frame["reference_group_id"].tolist()
        groups: dict[object, list[int]] = {}
        for index, group in enumerate(group_values):
            groups.setdefault(group, []).append(index)
        for column in column_ids:
            name = layout.column_names[column]
            if any(
                len({normalize_key(frame[name].iloc[index]) for index in indexes if not _blank(frame[name].iloc[index])}) > 1
                for indexes in groups.values()
            ):
                raise NotEvaluableError(
                    "Источник КМ меняется внутри dialogue и не является одним dialogue score",
                    column_id=column,
                    repair_hint=(
                        "Если документы задают score каждой строки или реплики, "
                        "используйте assessment_mode=qa или turn_with_history"
                    ),
                )
        if weighted and any(
            len({decimal_value(frame["input_query_count"].iloc[index]) for index in indexes}) > 1
            for indexes in groups.values()
        ):
            raise NotEvaluableError("Weight не константен внутри dialogue")
        any_missing = any(
            all(_blank(frame[layout.column_names[column]].iloc[index]) for index in indexes)
            for indexes in groups.values()
            for column in column_ids
        )
    else:
        any_missing = any(
            _blank(value)
            for column in column_ids
            for value in frame[layout.column_names[column]].tolist()
        )
    missing_policy = score["missing_policy"]
    if any_missing and missing_policy == "fail":
        raise NotEvaluableError("В источниках КМ есть пропуски, а missing_policy=fail")
    if any_missing and missing_policy != "fail" and not evidence["missing_policy"]:
        raise NotEvaluableError("Политика пропусков влияет на данные, но не доказана документами")

    denominator = score["majority_denominator"]
    if score["method"] == "majority":
        if any_missing and denominator is None:
            raise NotEvaluableError("При неполных голосах majority требует явный denominator")
        if any_missing and not evidence["denominator"]:
            raise NotEvaluableError("Denominator majority влияет на результат, но не имеет evidence")
        denominator = denominator or "declared"
    elif denominator is not None:
        raise MeasurementPlanError("majority_denominator допустим только для majority")

    if reducer == "frequency_weighted_mean":
        if layout.weight is None:
            raise NotEvaluableError("Взвешенная КМ требует физическую weight-колонку")
        weight_id = layout.weight["column_id"]
        if weight_id in layout.formula_rows:
            raise NotEvaluableError(
                "Кеш Excel-формулы нельзя использовать как frequency weight",
                rows=layout.formula_rows[weight_id],
            )
        if not evidence["reducer"]:
            raise NotEvaluableError("Взвешивание не доказано документами")

    release = proposal["release"]
    threshold_raw = release["threshold"]
    comparator = release["comparator"]
    if (threshold_raw is None) != (comparator is None):
        raise MeasurementPlanError("threshold и comparator должны быть заданы вместе или оба быть null")
    threshold = decimal_value(threshold_raw) if threshold_raw is not None else None
    if threshold is not None:
        if not evidence["release"]:
            raise MeasurementPlanError("Declared threshold остался без evidence")
        matches_scale = any(
            _span_has_value(spans[span_id][1], threshold, release["scale"])
            for span_id in evidence["release"]
        )
        alternative = {"ratio": "percent", "percent": "ratio"}.get(release["scale"])
        if not matches_scale and alternative and any(
            _span_has_value(spans[span_id][1], threshold, alternative)
            for span_id in evidence["release"]
        ):
            release["scale"] = alternative
            matches_scale = True
        if not matches_scale:
            raise NotEvaluableError("Threshold не найден в cited evidence", threshold=str(threshold))

    state = proposal["reported_value_state"]
    reported = proposal["reported_value"]
    if state == "ambiguous":
        validation_spans = {
            span_id: text
            for span_id, (document_id, text) in spans.items()
            if document_id == roles["validation_report"]
        }
        reported_candidates = _reported_reducer_candidates(
            validation_spans,
            reducer,
            release["scale"],
        )
        required_reported_value = reported_candidates[0] if len(reported_candidates) == 1 else None
        raise NotEvaluableError(
            "Validation report содержит неоднозначное итоговое значение КМ",
            reducer=reducer,
            required_reported_value=required_reported_value,
            candidate_spans=_reducer_candidate_spans(validation_spans),
        )
    if (state == "unambiguous") != (reported is not None):
        raise MeasurementPlanError("reported_value_state не согласован с reported_value")
    reported_value = reported_raw = reported_span = None
    if reported is not None:
        reported_value = decimal_value(reported["value"])
        reported_raw = reported["raw"]
        reported_span = reported["span_id"]
        if reported_span not in spans or spans[reported_span][0] != roles["validation_report"]:
            candidates = [
                span_id for span_id, (document_id, text) in spans.items()
                if document_id == roles["validation_report"]
                and _span_has_value(text, reported_value, release["scale"])
            ]
            raise NotEvaluableError(
                "Итоговая КМ должна ссылаться на span validation report",
                reported_span=reported_span,
                validation_report_document=roles["validation_report"],
                candidate_spans=candidates[:8],
            )
        span_text = spans[reported_span][1]
        if not _span_has_value(span_text, reported_value, release["scale"]):
            candidates = [
                span_id for span_id, (document_id, text) in spans.items()
                if document_id == roles["validation_report"]
                and _span_has_value(text, reported_value, release["scale"])
            ]
            suffix = f"; numeric candidate spans: {candidates[:8]}" if candidates else ""
            raise NotEvaluableError(
                f"reported_value не соответствует числу в cited span{suffix}"
            )
        if reported_span not in evidence["reported_value"]:
            raise MeasurementPlanError("reported_value span должен входить в evidence.reported_value")
        if release["scale"] == "percent":
            canonical = _percent_domain_value(span_text, reported_value)
            if canonical is not None:
                reported_value = canonical

    return MeasurementPlan(
        basket_id=context.basket_id,
        metric_name=proposal["metric_name"],
        document_roles=dict(roles),
        assessment_mode=assessment_mode,
        method=method,
        sources=tuple(sources),
        missing_policy=missing_policy,
        majority_denominator=denominator,
        reducer=reducer,
        threshold=threshold,
        comparator=comparator,
        scale=release["scale"],
        precision=release["precision"],
        reported_value=reported_value,
        reported_raw=reported_raw,
        reported_span_id=reported_span,
        evidence=evidence,
    )
