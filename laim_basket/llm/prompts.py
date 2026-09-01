"""Промпты двух LLM-задач: контекст (документы, снимок) -> задача -> схема.

Каждый документ обёрнут в XML-тег с ролью из имени порта; задача и схема
стоят в конце user-сообщения (рекомендация для длинных контекстов).
"""
from __future__ import annotations

import json

from .. import defaults
from .schemas import LAYOUT_SCHEMA, METRIC_SCHEMA

_LAYOUT_SYSTEM = """Ты размечаешь эталонную корзину GenAI-агента: одна XLSX-книга \
и три документа (отчёт о валидации, отчёт о разработке, инструкция ассессора).
Правила:
- Роли колонок задавай адресами колонок листа (A, B, C — как в Excel).
- Выбирай лист с пооценочными данными агента; справочные листы не выбирай.
- Опирайся на документы: отчёт о разработке часто содержит расшифровку колонок \
корзины — выпиши опорные фрагменты в quotes дословно.
- input_query — колонка запроса пользователя. Если весь диалог упакован в одну \
ячейку строки, задай grouping.kind=blob_row, dialogue_blob (та же колонка, \
маркеры реплик, container) и output_answer=null. Маркеры бери ДОСЛОВНО из \
dialogue_markers снимка (АГЕНТ и АГЕНТЫ — разные маркеры), без двоеточия.
- Группировка должна быть видна в данных (vertical merge, повторяющийся ключ, \
blob); один только session_id не делает корзину диалоговой.
- Если ответ агента разложен по двум колонкам (основная ветка + fallback), \
используй output_answer.coalesce.
- weight_column — только физическая колонка целых частот (freq, количество \
повторов); иначе null.
- Пустые ячейки ответов допустимы: их обработает политика пропусков плана."""

_METRIC_SYSTEM = """Ты определяешь ключевую метрику (КМ) корзины GenAI-агента \
и порядок её расчёта по колонкам канона.
Правила:
- Приоритет источника для имени и метода КМ: отчёт о валидации; затем отчёт о \
разработке; затем сама корзина (порог релиза — в отчёте о валидации).
- reported_value берётся ТОЛЬКО из отчёта о валидации; raw — дословная цитата \
числа из него ("93%", "0,9736"), код сверяет её с текстом отчёта. Значения в \
отчёте о валидации нет -> state=not_declared, даже если число есть в других \
документах. Несколько разных кандидатов без возможности выбрать -> \
state=ambiguous.
- Режим оценки (qa/dialogue/turn_with_history) не выбирай: его определяет \
физическая форма данных.
- sources: колонки по column_id из инвентаря; role: final_score | criterion | \
assessor_vote | prediction | target. Для prediction/target normalization=label; \
текстовые оценки задавай value_map-объектом в normalization.
- reducer=frequency_weighted_mean только когда документы говорят о взвешивании \
по частоте и в корзине есть weight-колонка.
- В quotes процитируй фрагменты документов, подтверждающие метрику, редьюсер и \
политику пропусков."""


def _document_block(document: dict) -> str:
    body = "\n".join(
        f"p{index:03d}: {text}"
        for index, text in enumerate(document["paragraphs"], start=1)
    )[: defaults.DOCUMENT_CHAR_CAP]
    return f"<{document['port']}>\n{body}\n</{document['port']}>"


def _documents_context(documents: tuple[dict, ...]) -> str:
    return "\n\n".join(_document_block(document) for document in documents)


def _dump(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False)


def layout_messages(evidence: dict, documents: tuple[dict, ...],
                    pinned_sheet: str, rejected_sheets: frozenset[str]) -> list[dict]:
    task = ["Определи разметку книги по правилам системного сообщения."]
    if pinned_sheet:
        task.append(f"Оператор закрепил лист {pinned_sheet!r} — выбери именно его.")
    if rejected_sheets:
        task.append(
            "Эти листы уже отвергнуты, КМ на них не строится: "
            f"{sorted(rejected_sheets)}. Выбери другой лист."
        )
    user = "\n\n".join([
        "ДОКУМЕНТЫ ПАКЕТА:",
        _documents_context(documents),
        "СНИМОК КНИГИ (структура и статистика, не данные):",
        _dump(evidence),
        " ".join(task),
        f"JSON SCHEMA:\n{_dump(LAYOUT_SCHEMA)}",
        "Верни только JSON по схеме.",
    ])
    return [{"role": "system", "content": _LAYOUT_SYSTEM},
            {"role": "user", "content": user}]


def metric_messages(column_inventory: list[dict], documents: tuple[dict, ...],
                    assessment_facts: dict) -> list[dict]:
    user = "\n\n".join([
        "ДОКУМЕНТЫ ПАКЕТА:",
        _documents_context(documents),
        "ИНВЕНТАРЬ КОЛОНОК КАНОНА:",
        _dump(column_inventory),
        "ФИЗИЧЕСКАЯ ФОРМА КОРЗИНЫ (определена кодом, не выбирается):",
        _dump(assessment_facts),
        "Определи ключевую метрику и порядок её расчёта.",
        f"JSON SCHEMA:\n{_dump(METRIC_SCHEMA)}",
        "Верни только JSON по схеме.",
    ])
    return [{"role": "system", "content": _METRIC_SYSTEM},
            {"role": "user", "content": user}]
