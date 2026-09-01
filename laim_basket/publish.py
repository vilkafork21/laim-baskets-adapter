"""Проекция внутреннего канона в формат тестового датасета.

Публикуется ровно один из двух листов спецификации: «Вариант для отд. запросов»
(строка = запрос/реплика) или «Вариант для диалога» (строка = сессия, реплики
упакованы в `dialogue`). Лист выбирается по канонизированной единице оценки,
которую адаптер выводит из физической формы входных данных. Сырые колонки
корзины в публикацию не попадают: каждая либо получила каноническое имя, либо
остаётся только в debug-отчётах.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .errors import NotEvaluableError
from .metric.engine import source_values
from .models import MeasurementPlan, ResolvedLayout
from .transform.values import blank, slug

FLAT_SHEET = "Вариант для отд. запросов"
DIALOGUE_SHEET = "Вариант для диалога"

_METRIC_ROLES = {"final_score", "criterion", "assessor_vote"}
_LABEL_ROLES = {"prediction": "output_answer", "target": "reference_answer"}
# Колонки спецификации с фиксированным местом; остальные (reference_answer,
# [module]_*, *_metric) идут между ними в физическом порядке колонок корзины.
_HEAD = (
    "solution_version",
    "scenario",
    "session_id",
    "query_id",
    "input_query_count",
    "input_query",
    "output_answer",
)
_TAIL = ("main_metric", "assessor_id")


@dataclass(frozen=True)
class PublishedUmr:
    """Опубликованная корзина: лист спецификации и карта имён для контракта.

    `published_columns` — column_id корзины -> опубликованное имя; по ней
    monitoring_metric называет источники КМ (в отличие от
    `ResolvedLayout.column_names`, где лежат сырые имена колонок).
    """

    variant: str
    sheet_name: str
    frame: pd.DataFrame
    published_columns: dict[str, str]
    dropped_non_constant: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "sheet_name": self.sheet_name,
            "columns": list(self.frame.columns),
            "published_columns": self.published_columns,
            "rows": len(self.frame),
            "dropped_non_constant": list(self.dropped_non_constant),
        }


def metric_slug(header: str) -> str:
    name = slug(header)
    return name if name.endswith("_metric") else f"{name}_metric"


def _letter(layout: ResolvedLayout, source: str) -> str:
    return layout.region.column_letters[layout.region.columns.index(source)]


def _header(layout: ResolvedLayout, column_id: str) -> str:
    return layout.region.columns[layout.region.column_letters.index(column_id)]


def _role_names(layout: ResolvedLayout) -> dict[str, str]:
    """column_id -> каноническое имя для колонок, получивших роль в layout."""
    roles = layout.roles
    names = {
        column_id: "solution_version"
        for column_id, name in layout.column_names.items()
        if name == "solution_version"
    }
    for role in ("input_query", "query_id", "session_id", "scenario", "assessor_id"):
        value = roles.get(role)
        if isinstance(value, dict) and "source" in value:
            names[_letter(layout, value["source"])] = role
    output = roles.get("output_answer")
    if isinstance(output, dict):
        for source in output.get("coalesce", [output.get("source")]):
            names[_letter(layout, source)] = "output_answer"
    for index, source in enumerate(roles.get("reference_answers", []), start=1):
        names[_letter(layout, source)] = "reference_answer" if index == 1 else f"reference_answer_{index}"
    if layout.weight:
        names[_letter(layout, layout.weight["source"])] = "input_query_count"
    if layout.grouping["kind"] == "column":
        names[_letter(layout, layout.grouping["column"])] = "session_id"
    return names


def published_names(layout: ResolvedLayout, plan: MeasurementPlan | None) -> list[tuple[str, str]]:
    """Пары (column_id, каноническое имя) для всех публикуемых колонок.

    Одна физическая колонка может дать две: роль layout и источник КМ. Источники
    score/criterion/vote получают `<slug>_metric`, prediction/target —
    `[module]_output_answer`/`[module]_reference_answer` — собственное имя нужно
    даже на колонке ответа, потому что в monitoring-ветке предсказание приходит
    отдельным полем (route_label из трейса), а `output_answer` несёт текст ответа.
    """
    roles = _role_names(layout)
    result = list(roles.items())
    if plan is None:
        return result
    published = set(roles.values())
    for source in plan.sources:
        column_id, role = source["column_id"], source["role"]
        header = _header(layout, column_id)
        if role in _LABEL_ROLES:
            name = f"{slug(header)}_{_LABEL_ROLES[role]}"
        elif column_id in roles:
            raise NotEvaluableError(
                "Источник КМ занят канонической ролью и не может быть метрикой",
                column_id=column_id, role=role, canonical_role=roles[column_id],
            )
        else:
            name = metric_slug(header)
        if name in published or name == "main_metric":
            raise NotEvaluableError(
                "Имя колонки источника КМ конфликтует с другой опубликованной колонкой",
                column_id=column_id, column_name=name,
            )
        published.add(name)
        result.append((column_id, name))
    return result


def _source_columns(
    frame: pd.DataFrame, layout: ResolvedLayout, plan: MeasurementPlan, names: dict[str, str]
) -> dict[str, list[object]]:
    """Колонки источников КМ: метрики — нормализованными числами, module-level
    prediction/target — как есть (роли layout уже присутствуют в frame)."""
    metrics = source_values(frame, layout, plan)
    result: dict[str, list[object]] = {}
    for source in plan.sources:
        name = names[source["column_id"]]
        if source["role"] in _METRIC_ROLES:
            result[name] = [
                None if value is None else float(value) for value in metrics[source["column_id"]]
            ]
        elif name not in frame:
            result[name] = frame[layout.column_names[source["column_id"]]].tolist()
    return result


def _session_values(frame: pd.DataFrame, layout: ResolvedLayout) -> list[object]:
    """session_id строки: явная роль, группирующая колонка либо номер группы от 1."""
    groups = frame["session_id"].tolist()
    session_role = layout.roles.get("session_id")
    if isinstance(session_role, dict):
        source = session_role["source"]
    elif layout.grouping["kind"] == "column":
        source = layout.grouping["column"]
    else:
        return [int(group) + 1 for group in groups]
    raw = frame[layout.column_names[_letter(layout, source)]].tolist()
    first_value: dict[object, object] = {}
    for group, value in zip(groups, raw):
        if group not in first_value and not blank(value):
            first_value[group] = value
    return [
        first_value[group] if group in first_value else int(group) + 1
        for group in groups
    ]


def _spec_columns(names: list[tuple[str, str]]) -> list[str]:
    """Колонки спецификации в порядке публикации (без dialogue).

    Колонки без фиксированного места (reference_answer, `[module]_*`, `*_metric`)
    идут в физическом порядке колонок корзины: Z перед AA.
    """
    ordered = sorted(names, key=lambda item: (len(item[0]), item[0]))
    middle = [name for _column_id, name in ordered if name not in _HEAD and name not in _TAIL]
    return [*_HEAD, *middle, *_TAIL]


def _flat(frame: pd.DataFrame, layout: ResolvedLayout, names: list[tuple[str, str]]) -> pd.DataFrame:
    result = frame.copy()
    if "reference_group_id" in result or isinstance(layout.roles.get("session_id"), dict):
        result["session_id"] = _session_values(result, layout)
    else:
        result = result.drop(columns=["session_id"])
    return result[[name for name in _spec_columns(names) if name in result]]


def _dialogue_turns(frame: pd.DataFrame, positions: list[int]) -> str:
    ordered = sorted(positions, key=lambda position: frame["turn_index"].iloc[position])
    turns = []
    for position in ordered:
        query, answer = frame["input_query"].iloc[position], frame["output_answer"].iloc[position]
        turns.append((
            str(frame["query_id"].iloc[position]),
            "" if blank(query) else str(query),
            "" if blank(answer) else str(answer),
        ))
    return repr(turns)


def _dialogue(
    frame: pd.DataFrame, layout: ResolvedLayout, names: list[tuple[str, str]]
) -> tuple[pd.DataFrame, list[str]]:
    """Одна строка на сессию; сессионная колонка, меняющаяся внутри диалога, опускается."""
    session = frame.assign(session_id=_session_values(frame, layout))
    # На диалоговом листе строка — сессия: turn-колонки уходят в `dialogue`,
    # остаются только те, что несут одно значение на весь диалог.
    session_columns = [
        name for name in _spec_columns(names)
        if name in session
        and name not in ("query_id", "input_query_count", "input_query", "output_answer")
    ]
    positions: dict[object, list[int]] = {}
    for position, group in enumerate(session["reference_group_id"].tolist()):
        positions.setdefault(group, []).append(position)
    records = []
    dropped: set[str] = set()
    for indexes in positions.values():
        record: dict[str, object] = {"dialogue": _dialogue_turns(session, indexes)}
        for name in session_columns:
            values = [session[name].iloc[position] for position in indexes]
            present = [value for value in values if not blank(value)]
            if len({str(value) for value in present}) > 1:
                dropped.add(name)
            record[name] = present[0] if present else None
        records.append(record)
    head = [
        name
        for name in session_columns
        if name in ("solution_version", "scenario", "session_id")
    ]
    order = [*head, "dialogue", *[name for name in session_columns if name not in head]]
    result = pd.DataFrame(records, dtype=object)
    return result[[name for name in order if name not in dropped]], sorted(dropped)


def publish_umr(frame: pd.DataFrame, layout: ResolvedLayout, plan: MeasurementPlan | None) -> PublishedUmr:
    """Спроецировать внутренний канон (с сырыми колонками) в лист спецификации."""
    names = published_names(layout, plan)
    # Источник КМ добавляется после роли того же column_id, поэтому в карте
    # контракта побеждает имя источника — его и называет monitoring_metric.
    contract = dict(names)
    source = frame.copy()
    if plan is not None:
        for name, values in _source_columns(frame, layout, plan, contract).items():
            source[name] = values
    dropped: list[str] = []
    if plan is not None and plan.assessment_mode == "dialogue":
        variant, sheet_name = "dialogue", DIALOGUE_SHEET
        published, dropped = _dialogue(source, layout, names)
    else:
        variant, sheet_name = "flat", FLAT_SHEET
        published = _flat(source, layout, names)
    published = published.reset_index(drop=True)
    if "main_metric" in published:
        published["main_metric"] = [
            None if blank(value) else float(value) for value in published["main_metric"].tolist()
        ]
    return PublishedUmr(variant, sheet_name, published, contract, tuple(dropped))
