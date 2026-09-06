"""Узкие промпты: сначала физический layout, затем семантика измерения."""

import json

from .. import defaults
from ..contracts import LAYOUT_SCHEMA, MEASUREMENT_SCHEMA
from ..errors import NotEvaluableError


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1)


def _bounded(value: object, label: str) -> str:
    text = _dump(value)
    if len(text) > defaults.EVIDENCE_PROMPT_CHAR_CAP:
        raise NotEvaluableError(
            f"{label} не помещается в безопасный LLM-контекст без усечения",
            size=len(text), limit=defaults.EVIDENCE_PROMPT_CHAR_CAP,
        )
    return text


LAYOUT_SYSTEM = """Ты определяешь только физическую структуру XLSX-корзины.
Верни один JSON по схеме. Не определяй формулу ключевой метрики и не отбрасывай
строки. first_data_row, last_data_row и footer модель не задает: их докажет код.
Колонки адресуются физическим column_id и дословным header. Выбери один лист,
а все остальные перечисли в ignored_sheets. Если данные есть на нескольких листах,
выбери тот, который оценивают документы этой корзины: DOCUMENTS называет агента,
единицу оценки и колонки разметки. Одна книга может содержать листы разных агентов. Не включай первую строку данных в
header_rows: дополнительный уровень шапки допустим только при физическом
horizontal merge родительского уровня. grouping описывает только физическое
представление dialogue. grouping.column заполняется только для kind=column;
dialogue_blob заполняется только для kind=blob_row. kind=merged_rows допустим
только при вертикальном merge ячеек в строках данных; горизонтальные merge
многоуровневой шапки его не подтверждают. Повторяющийся обычный dialogue/session
ID задаётся как kind=column. kind=blob_row допустим только когда вся строка хранит
диалог в одной blob-ячейке: input_query обязан указывать на ту же колонку,
что и dialogue_blob, а output_answer должен быть null. roles.session_id указывает
только на явную физическую колонку идентификатора сессии; для blob_row сохрани её,
если она присутствует, иначе верни null. В blob_row query_id реплик берётся из
официальных троек либо синтезируется при разборе marker blob, поэтому отдельную
колонку session ID не назначай query_id. Не угадывай session_id по значениям.
Если строка уже содержит явные отдельные колонки input_query и output_answer,
сохрани эти роли, выбери kind=column для повторяющегося dialogue/session ID
или kind=none без него и задай dialogue_blob=null, даже если есть
вспомогательная колонка с полной историей. Если основной столбец ответа заполнен
только для части ветвей, а другой столбец содержит действие, маршрут или иной
результат агента для оставшихся строк, задай output_answer={"coalesce":[...]}
в порядке от содержательного ответа к fallback-результату. Пересечение заполненных
строк допустимо: используется первое непустое значение. Coalesce допустим только
по наблюдаемому построчному заполнению, не объединяет тексты, а его источники нельзя
повторно назначать scenario или другой ролью. Одна физическая колонка не может
одновременно быть
output, scenario или reference. Никаких значений,
порогов или Python-выражений. Weight указывай только для физической колонки
положительных целых frequency/count, но не для score или criterion."""


def document_intro(documents: object) -> list[dict[str, object]]:
    """Начало каждого документа: чей это агент и что размечали.

    Полный корпус в layout-промпт не помещается и не нужен — семантику метрики
    определяет второй вызов; здесь достаточно вводной части, чтобы выбрать лист.
    """
    intro = []
    for document in documents:
        spans = list(document["spans"])[: defaults.DOCUMENT_INTRO_SPANS]
        text = " ".join(str(span["text"]) for span in spans)
        intro.append({
            "id": document["id"],
            "file_name": document["file_name"],
            "text": text[: defaults.DOCUMENT_INTRO_CHAR_CAP],
        })
    return intro


def layout_messages(
    basket_id: str,
    evidence: object,
    documents: object,
    rejected_sheets: tuple[str, ...] = (),
    pinned_sheet: str = "",
) -> list[dict[str, str]]:
    constraints = ""
    if pinned_sheet:
        constraints += f"ОБЯЗАТЕЛЬНЫЙ ЛИСТ (задан оператором workflow): {_dump(pinned_sheet)}\n\n"
    if rejected_sheets:
        constraints += f"ЗАПРЕЩЁННЫЕ ЛИСТЫ (на них КМ уже не построилась):\n{_dump(list(rejected_sheets))}\n\n"
    user = (
        f"basket_id: {basket_id}\n\nDOCUMENTS:\n{_bounded(document_intro(documents), 'document intro')}\n\n"
        f"WORKBOOK:\n{_bounded(evidence, 'workbook evidence')}\n\n{constraints}"
        f"JSON SCHEMA:\n{_dump(LAYOUT_SCHEMA)}\n\nВерни layout JSON."
    )
    return [{"role": "system", "content": LAYOUT_SYSTEM}, {"role": "user", "content": user}]


MEASUREMENT_SYSTEM = """Ты извлекаешь из трёх документов (инструкция ассессора, отчёт о разработке,
отчёт о валидации) определение ключевой метрики (КМ) агента и записываешь его
как формулу над колонками корзины. Верни один JSON по закрытой схеме. Не вычисляй
значение и не пиши код. Используй только перечисленные column_id и span IDs;
каждое решение подтверждай spans.

inputs — колонки корзины, которые участвуют в формуле. judged=true, если колонку
проставлял разметчик (итог, критерий, голос, истинный класс): на мониторинге её
воспроизведёт LLM-судья. judged=false — ответ агента (класс, маршрут), он
наблюдается в трейсах. Имена входов — короткие идентификаторы (prediction, target,
полнота, оценка).

formula — как метрика определена в отчёте, над именами inputs и weight
(weight — положительная частота/count из layout, использовать только если отчёт
явно называет агрегацию взвешенной). Язык: + - * /, сравнения == != < <= > >=,
and/or/not; агрегаты mean(x), wmean(x, weight), sum(x), count(x); построчные
avg(a, b, ...), min(...), max(...), abs(x), fillna(x, число),
majority(голоса..., declared=True|False); по классам precision(prediction, target,
average), recall(...), f1(...), где average — "macro", "micro", "weighted" или метка
класса. Пропуски агрегаты пропускают; если отчёт считает их нулём — fillna явно.
Примеры:
  mean(итог)                          доля успешных (готовая оценка разметчика)
  mean(prediction == target)          accuracy
  wmean(prediction == target, weight) взвешенная accuracy
  mean((полнота + точность) / 2)      среднее критериев
  mean(min(полнота, точность))        все критерии выполнены
  mean(majority(а1, а2, а3))          решение большинства асессоров
  f1(prediction, target, "macro")     macro-F1
  mean(оценка >= 4)                   доля с оценкой не ниже порога
Формула даёт одно число в долях (release.scale=percent означает, что отчёт
публикует его в процентах); она обязана воспроизвести reported_value на корзине —
это проверяется пересчётом, при расхождении ты получишь repair с диагностикой.

assessment_mode — единица оценки только по документам: qa для независимого
запрос-ответ, turn_with_history для реплики с учётом предыдущих, dialogue для одной
оценки всего разговора. Физическая форма корзины (merged rows, blob) может
скорректировать выбор — это сделает код.

Validation report — источник заявленной КМ: если итоговое значение указано
однозначно, верни reported_value с точным raw и span; если значений несколько и
целевое не названо — reported_value_state=ambiguous. Значение нельзя брать из
XLSX. Порог (release.threshold) информационный: если его нет в документах —
threshold=null, comparator=null."""


def measurement_messages(
    basket_id: str,
    documents: object,
    columns: object,
) -> list[dict[str, str]]:
    user = (
        f"basket_id: {basket_id}\n\nDOCUMENT CORPUS:\n{_bounded(documents, 'document corpus')}\n\n"
        f"AVAILABLE COLUMNS:\n{_bounded(columns, 'column inventory')}\n\n"
        f"JSON SCHEMA:\n{_dump(MEASUREMENT_SCHEMA)}\n\nВерни MeasurementPlan JSON."
    )
    return [{"role": "system", "content": MEASUREMENT_SYSTEM}, {"role": "user", "content": user}]
