"""Два ограниченных structured-вызова LLM с python-валидацией в repair-цикле."""

import copy
import logging
import re
from decimal import Decimal

from .. import defaults
from ..contracts import LAYOUT_SCHEMA, MEASUREMENT_SCHEMA
from ..errors import (
    LayoutError,
    NotEvaluableError,
)
from ..layout import resolve_layout
from ..measurement import reported_quantum, resolve_measurement_plan
from ..metric.engine import evaluate
from ..publish import publish_umr
from ..transform.canon import build_canon
from ..transform.grouping import apply_grouping
from ..transform.values import blank
from .prompts import layout_messages, measurement_messages
from .structured import request_structured


logger = logging.getLogger(__name__)

_BLOB_MARKER = re.compile(defaults.BLOB_MARKER_PATTERN)


def _alternating_marker_pair(samples: list[str]) -> tuple[str, str] | None:
    pair = None
    for sample in samples:
        sequence = _BLOB_MARKER.findall(sample)
        if len(sequence) < 2 or len(sequence) % 2:
            return None
        current = (sequence[0], sequence[1])
        if current[0] == current[1] or any(
            tuple(sequence[index:index + 2]) != current
            for index in range(0, len(sequence), 2)
        ):
            return None
        if pair is not None and pair != current:
            return None
        pair = current
    return pair


def _repeated_inside_physical_group(layout, blob_column: str) -> bool:
    """Полная история внутри нескольких turn-строк не является blob-row."""
    blob_index = layout.region.column_letters.index(blob_column)
    for group_index in range(len(layout.region.columns)):
        if group_index == blob_index:
            continue
        positions: dict[str, list[int]] = {}
        for row_index, row in enumerate(layout.region.rows):
            if blank(row[group_index]):
                continue
            positions.setdefault(str(row[group_index]).strip(), []).append(row_index)
        repeated = [indexes for indexes in positions.values() if len(indexes) > 1]
        if repeated and all(
            indexes == list(range(indexes[0], indexes[-1] + 1))
            and len({str(layout.region.rows[index][blob_index]) for index in indexes}) == 1
            for indexes in repeated
        ):
            return True
    return False


def _blob_row_candidates(layout, evidence: dict[str, object]) -> list[dict[str, object]]:
    """Однозначные blob-источники для уже выбранного физического input_query."""
    input_role = layout.roles.get("input_query")
    if not isinstance(input_role, dict):
        return []
    input_source = input_role.get("source")
    input_columns = {
        column_id
        for column_id, source in zip(
            layout.region.column_letters, layout.region.columns
        )
        if source == input_source
    }
    sheets = evidence.get("sheets") if isinstance(evidence, dict) else None
    if not isinstance(sheets, list):
        return []
    sheet = next((
        item for item in sheets
        if isinstance(item, dict) and item.get("sheet_name") == layout.sheet_name
    ), None)
    if sheet is None:
        return []
    columns = {
        item.get("column_id"): item
        for item in sheet.get("columns", [])
        if isinstance(item, dict)
    }
    candidates = []
    for hint in sheet.get("dialogue_hints", []):
        if not isinstance(hint, dict) or hint.get("column_id") not in input_columns:
            continue
        column_id = str(hint["column_id"])
        column = columns.get(column_id)
        if not isinstance(column, dict):
            continue
        samples = [str(value) for value in column.get("samples", []) if str(value).strip()]
        pair = _alternating_marker_pair(samples)
        if pair is None:
            continue
        if _repeated_inside_physical_group(layout, column_id):
            continue
        candidates.append({
            "column_id": column_id,
            "source": input_source,
            "markers": list(pair),
            "container": (
                "python_list"
                if all(value.lstrip().startswith("[") for value in samples)
                else "plain_text"
            ),
            "required_changes": {
                "grouping": {"kind": "blob_row", "column": None},
                "input_query_and_dialogue_blob_column": (
                    f"same physical address {column_id}"
                ),
                "output_answer": None,
            },
        })
    return candidates


def _prove_flat_output(
    layout, frame, evidence: dict[str, object], confirmed_reason: str = ""
) -> None:
    """Flat-layout не может молча потерять результаты ветвей агента."""
    if layout.grouping["kind"] == "blob_row":
        return
    missing = [
        index for index, value in enumerate(frame["output_answer"].tolist())
        if blank(value)
    ]
    if not missing:
        return
    if confirmed_reason:
        logger.warning(
            "Flat output_answer: %d пустых ответов подтверждены моделью как "
            "свойство данных (%s) — публикуются как есть под missing_policy",
            len(missing), confirmed_reason,
        )
        return

    output = layout.roles.get("output_answer")
    output_sources = (
        output["coalesce"]
        if isinstance(output, dict) and "coalesce" in output
        else [output["source"]]
        if isinstance(output, dict) and "source" in output
        else []
    )
    excluded = set(output_sources)
    for role in ("input_query", "query_id", "session_id", "assessor_id"):
        value = layout.roles.get(role)
        if isinstance(value, dict) and "source" in value:
            excluded.add(value["source"])
    excluded.update(layout.roles.get("reference_answers", []))
    if layout.grouping.get("column"):
        excluded.add(layout.grouping["column"])
    if layout.weight:
        excluded.add(layout.weight["source"])

    candidates = []
    for column_id, source in zip(
        layout.region.column_letters, layout.region.columns
    ):
        if source in excluded:
            continue
        raw_name = layout.column_names[column_id]
        values = frame[raw_name].tolist()
        if not all(not blank(values[index]) for index in missing):
            continue
        samples = [str(values[index])[:120] for index in missing[:3]]
        candidates.append({
            "column_id": column_id,
            "column": source,
            "samples_on_missing_rows": samples,
        })
    blob_candidates = _blob_row_candidates(layout, evidence)
    raise LayoutError(
        "Flat output_answer теряет результаты: в выбранном источнике есть пустые строки",
        output_sources=output_sources,
        missing_output_count=len(missing),
        missing_source_rows=[int(frame["source_row_id"].iloc[index]) for index in missing[:10]],
        fallback_candidates=candidates[:12],
        blob_row_candidates=blob_candidates,
        repair_hint=(
            "Выбери ровно одну доказанную ветку. Для flat-строк с результатами "
            "разных ветвей задай output_answer.coalesce: текущий содержательный "
            "ответ первым, семантический fallback из fallback_candidates вторым. "
            "Если blob_row_candidates содержит единственный полный диалог, задай "
            "grouping.kind=blob_row, input_query и dialogue_blob.column на его общий "
            "адрес, а output_answer=null. Не выбирай ID, score или metadata только "
            "ради заполненности. Если ни один кандидат не является семантическим "
            "ответом, верни тот же layout, добавив в evidence ключ output_gaps с "
            "кратким обоснованием — пустые ответы опубликуются честно и попадут "
            "под missing_policy плана."
        ),
    )


def _repair_proven_blob_layout(
    proposal: dict[str, object], error: LayoutError
) -> dict[str, object] | None:
    candidates = error.details.get("blob_row_candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        return None
    candidate = candidates[0]
    input_address = proposal["roles"]["input_query"]
    if input_address["column"] != candidate["column_id"]:
        return None
    repaired = copy.deepcopy(proposal)
    repaired["roles"]["output_answer"] = None
    repaired["grouping"] = {"kind": "blob_row", "column": None}
    repaired["dialogue_blob"] = {
        "column": copy.deepcopy(input_address),
        "container": candidate["container"],
        "question_marker": candidate["markers"][0],
        "answer_marker": candidate["markers"][1],
    }
    repaired.setdefault("evidence", {})["deterministic_layout_repair"] = (
        "Flat-ответ неполон; выбранная входная колонка — самостоятельный диалог "
        "с чередующимися маркерами, подтверждённый физическими данными XLSX"
    )
    return repaired


def generate_layout(client, context, evidence, rejected_sheets=(), pinned_sheet=""):
    resolved = {}
    recoverable = {}

    def materialize(proposal: dict):
        layout = resolve_layout(proposal, context)
        sheet = context.sheets[layout.sheet_name]
        grouped = apply_grouping(sheet, layout.region, layout.transform_config())
        frame, conversion = build_canon(
            grouped, layout.region, layout.transform_config()
        )
        return layout, frame, conversion

    def validate(proposal: dict) -> None:
        # Канонизация и UMR-валидация внутри repair-петли: LLM получает
        # конкретные нарушения и может исправить layout вместо гибели рана.
        if pinned_sheet and proposal["sheet_name"] != pinned_sheet:
            raise LayoutError(
                "Оператор workflow задал лист явно: выбери именно его",
                sheet_name=proposal["sheet_name"],
                pinned_sheet=pinned_sheet,
            )
        if proposal["sheet_name"] in rejected_sheets:
            raise LayoutError(
                "Этот лист уже отвергнут: на нём КМ не строится, выбери другой",
                sheet_name=proposal["sheet_name"],
                rejected_sheets=list(rejected_sheets),
            )
        layout, frame, conversion = materialize(proposal)
        output_gaps = str((proposal.get("evidence") or {}).get("output_gaps") or "").strip()
        try:
            _prove_flat_output(layout, frame, evidence, output_gaps)
        except LayoutError as exc:
            repaired = _repair_proven_blob_layout(proposal, exc)
            if repaired is None:
                raise
            layout, frame, conversion = materialize(repaired)
            _prove_flat_output(layout, frame, evidence)
            proposal.clear()
            proposal.update(repaired)
        validation = conversion["umr_validation"]
        if validation["status"] != "passed":
            error = LayoutError(
                "Канонический UMR не прошёл валидацию",
                **validation,
                repair_hint=(
                    "Смотри source_rows нарушений: пустой input_query внутри "
                    "диапазона данных означает неверно выбранную колонку или "
                    "неучтённую структуру (merged_rows/blob_row) — выбери "
                    "источник, заполненный в каждой строке данных."
                ),
            )
            missing_values = validation["missing_required_values"]
            if (
                not validation["missing_required"]
                and set(missing_values) == {"input_query"}
                and not validation["type_violations"]
                and not validation["context_violations"]
                and "reference_group_id" not in frame
                and missing_values["input_query"]["count"] < len(frame)
            ):
                recoverable.update(
                    error=error,
                    proposal=copy.deepcopy(proposal),
                    layout=layout,
                    frame=frame,
                    conversion=conversion,
                )
            raise error
        resolved.update(layout=layout, frame=frame, conversion=conversion)

    try:
        proposal = request_structured(
            client,
            layout_messages(
                context.basket_id, evidence, context.documents, rejected_sheets, pinned_sheet
            ),
            LAYOUT_SCHEMA,
            "layout",
            validate_extra=validate,
            max_turns=5,
        )
    except LayoutError as exc:
        if recoverable.get("error") is not exc:
            raise
        frame = recoverable["frame"]
        keep = [not blank(value) for value in frame["input_query"].tolist()]
        dropped_rows = [
            int(value) for value in frame.loc[[not value for value in keep], "source_row_id"]
        ]
        conversion = copy.deepcopy(recoverable["conversion"])
        frame = frame.loc[keep].reset_index(drop=True)
        conversion["row_accounting"]["canon_rows"] = len(frame)
        conversion["row_accounting"]["dropped_blank_input_query_rows"] = dropped_rows
        conversion["umr_validation"]["status"] = "passed"
        conversion["umr_validation"]["missing_required_values"] = {}
        proposal = recoverable["proposal"]
        resolved.update(
            layout=recoverable["layout"], frame=frame, conversion=conversion
        )
    return proposal, resolved["layout"], resolved["frame"], resolved["conversion"]


def _matching_identity_plans(frame, layout, target: Decimal, tolerance: Decimal) -> list[dict]:
    """Identity-планы, которые воспроизводят заявленную КМ в пределах разряда."""
    import pandas as pd

    matches = []
    weights = pd.to_numeric(frame["input_query_count"], errors="coerce")
    for column_id, name in layout.column_names.items():
        if name not in frame:
            continue
        series = pd.to_numeric(frame[name], errors="coerce")
        if not series.notna().any():
            continue
        variants = [("qa", "mean", series.dropna().mean())]
        present = series.notna() & weights.notna()
        if layout.weight is not None and present.any() and weights[present].sum() > 0:
            variants.append((
                "qa",
                "frequency_weighted_mean",
                (series[present] * weights[present]).sum() / weights[present].sum(),
            ))
        if "reference_group_id" in frame:
            per_group = series.groupby(frame["reference_group_id"], sort=False).first()
            if per_group.notna().any():
                variants.append(("dialogue", "mean", per_group.dropna().mean()))
                if layout.weight is not None:
                    group_weights = weights.groupby(frame["reference_group_id"], sort=False).first()
                    present_groups = per_group.notna() & group_weights.notna()
                    if present_groups.any() and group_weights[present_groups].sum() > 0:
                        variants.append((
                            "dialogue",
                            "frequency_weighted_mean",
                            (per_group[present_groups] * group_weights[present_groups]).sum()
                            / group_weights[present_groups].sum(),
                        ))
        for unit, reducer, value in variants:
            if abs(Decimal(str(value)) - target) <= tolerance:
                matches.append({
                    "method": "identity",
                    "column_id": column_id,
                    "column": name,
                    "assessment_mode": unit,
                    "reducer": reducer,
                    "recomputed": float(value),
                })
    return matches[:12]


def _reconciliation_gate(plan, km: dict, frame, layout) -> None:
    """Отчёт о валидации — источник истины КМ: публикуется его значение.

    Расхождение пересчёта отправляется в repair только когда в корзине есть
    доказуемо сходящийся альтернативный identity-план (выбран не тот score);
    без такой альтернативы план принимается, а расхождение остаётся честной
    пометкой reconciliation="mismatch" в контракте (решение оператора контура:
    baseline по умолчанию — из отчёта о валидации).
    """
    reconciliation = km.get("reconciliation") or {}
    if plan.reported_value is None or reconciliation.get("status") != "mismatch":
        return
    difference = abs(Decimal(str(reconciliation.get("difference", 0))))
    tolerance = reported_quantum(plan)
    # Диагностика колонок идёт в ratio-домене; цель и разряд приводятся из
    # percent-домена, иначе физические 0/1 никогда не совпадут с 98.7.
    divisor = Decimal(100) if plan.scale == "percent" else Decimal(1)
    matches = _matching_identity_plans(
        frame, layout, plan.reported_value / divisor, tolerance / divisor
    )
    if not matches:
        logger.warning(
            "Пересчитанная КМ (%s) расходится с заявленной в validation report "
            "(%s); публикуется заявленная, reconciliation=mismatch",
            km.get("recomputed_value"), plan.reported_value,
        )
        return
    raise NotEvaluableError(
        "Пересчитанная КМ расходится с заявленной в validation report, а корзина "
        "содержит identity-план, воспроизводящий заявленную: используй ровно один "
        "доказанный plan из matching_identity_plans.",
        reported_value=str(plan.reported_value),
        recomputed_value=km.get("recomputed_value"),
        difference=str(difference),
        published_quantum=str(tolerance),
        matching_identity_plans=matches,
    )


def generate_measurement_plan(client, context, layout, frame, columns):
    resolved = {}

    def validate(proposal: dict) -> None:
        plan = resolve_measurement_plan(proposal, context, layout, frame)
        scored, km = evaluate(frame, layout, plan)
        _reconciliation_gate(plan, km, frame, layout)
        # Проекция в формат тестового датасета — часть валидации: конфликт имён
        # колонок возвращается модели как repair, а не роняет прогон.
        published = publish_umr(scored, layout, plan)
        resolved.update(proposal=proposal, plan=plan, published=published, km=km)

    request_structured(
        client,
        measurement_messages(context.basket_id, context.documents, columns),
        MEASUREMENT_SCHEMA,
        "measurement",
        validate_extra=validate,
        max_turns=5,
    )
    return (
        resolved["proposal"], resolved["plan"], resolved["published"], resolved["km"]
    )
