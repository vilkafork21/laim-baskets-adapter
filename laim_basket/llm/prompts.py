"""Промпты двух LLM-задач: контекст (документы, снимок) -> задача -> схема.

Каждый документ обёрнут в XML-тег с ролью из имени порта; задача и схема
стоят в конце user-сообщения (рекомендация для длинных контекстов).
"""
from __future__ import annotations

import json
import logging

from .. import defaults
from .schemas import LAYOUT_SCHEMA, METRIC_SCHEMA

logger = logging.getLogger(__name__)

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
dialogue_markers снимка (АГЕНТ и АГЕНТЫ — разные маркеры), без двоеточия. \
Если ячейка — JSON-массив сообщений с полями role/content или список пар \
(вопрос, ответ), укажи container=python_list, а маркерами — имена ролей \
клиента и агента как они записаны в данных (например user и assistant).
- Группировка должна быть видна в данных (vertical merge, повторяющийся ключ, \
blob); один только session_id не делает корзину диалоговой.
- Если ответ агента разложен по двум колонкам (основная ветка + fallback), \
используй output_answer.coalesce.
- assessor_id — только идентификатор разметчика, не его вердикт или оценка.
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
- assessment_mode задаёт методика: qa — отдельный ответ без истории, \
turn_with_history — ответ с предшествующим контекстом, dialogue — итог всего диалога. \
В quotes процитируй основание единицы. Merge, session_id и упаковка Excel не определяют её.
- evaluation — кандидат определения для однократного ревью: rubric описывает итоговую \
оценку main_metric, score_values задаёт шкалу независимо от встреченных значений; \
higher_is_better и defect_threshold задают направление и границу дефекта. \
required_evidence перечисляет обязательные факты, prediction_observable — наблюдаемое \
предсказание (не выводи route_label из названия accuracy). observation_profile требует \
проверки протокола трассировки при подключении. Эти предложения не являются допуском.
- normalization=numeric сохраняет числовую шкалу; percent явно делит процентные пункты \
на 100. Максимум колонки не является основанием для преобразования шкалы.
- sources: колонки по column_id из инвентаря; role: final_score | criterion | \
assessor_vote | prediction | target. Для prediction/target normalization=label; \
текстовые оценки задавай value_map-объектом в normalization.
- identity требует ровно один final_score; mean_criteria/all_criteria — \
минимум два criterion; majority/all_assessors — минимум два assessor_vote; \
accuracy — ровно по одному prediction и target.
- reducer=frequency_weighted_mean только когда документы говорят о взвешивании \
по частоте и в корзине есть weight-колонка.
- В quotes процитируй фрагменты документов, подтверждающие метрику, редьюсер и \
политику пропусков."""


def _document_block(document: dict) -> str:
    body = "\n".join(
        f"p{index:03d}: {text}"
        for index, text in enumerate(document["paragraphs"], start=1)
    )
    cap = defaults.DOCUMENT_CHAR_CAP
    if len(body) > cap:
        # Модель не увидит хвост документа; КМ в отчётах о валидации нередко
        # объявлена таблицей ближе к концу — срез обязан быть виден в логе.
        logger.warning(
            "Документ %s (%s) обрезан до %d символов из %d (%d абзацев): "
            "модель не видит его конец",
            document["port"], document["name"], cap, len(body), len(document["paragraphs"]),
        )
        body = body[:cap]
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
        "ФИЗИЧЕСКАЯ ФОРМА КОРЗИНЫ (не определяет семантическую единицу):",
        _dump(assessment_facts),
        "Определи ключевую метрику и порядок её расчёта.",
        f"JSON SCHEMA:\n{_dump(METRIC_SCHEMA)}",
        "Верни только JSON по схеме.",
    ])
    return [{"role": "system", "content": _METRIC_SYSTEM},
            {"role": "user", "content": user}]
