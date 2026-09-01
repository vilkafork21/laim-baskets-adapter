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


MEASUREMENT_SYSTEM = """Ты извлекаешь семантику одной ключевой метрики из трех
полных документных корпусов DOCX/TXT. Верни один JSON по закрытой схеме. Не вычисляй значение и
не пиши формулу или код. Используй только перечисленные column_id и span IDs.
Каждое решение подтверждай конкретными spans. Бизнес-параметры нельзя угадывать.
Если параметр влияет на наблюдаемые данные, он обязан иметь evidence.
Validation report имеет приоритет: если итоговая КМ указана однозначно, верни
reported_value с exact raw text и span; если значений несколько и связь неясна,
reported_value_state=ambiguous. Несколько значений сами по себе не означают
ambiguous, если отчет отдельно называет используемый целевой вариант. reported_value
нельзя брать или вычислять из XLSX: его value и raw должны совпадать с числом в
выбранном validation span. Numeric class IDs для accuracy запрещены, labels
сравниваются как labels. Formula-backed columns использовать как score нельзя.
Если source vertically_merged, assessment_mode должен быть dialogue. Threshold
информационный и не управляет публикацией; если его нет в документах, верни
threshold=null, comparator=null и не угадывай их.
assessment_mode выбирай только по документам: qa для независимого Q/A,
turn_with_history для отдельной оценки текущей реплики с предыдущими репликами,
dialogue для одной оценки всего разговора. Наличие session ID или группировки в
XLSX само по себе не определяет режим. dialogue допустим, только если каждый source
задает одно значение на dialogue и постоянен внутри его строк. Row/turn criteria,
votes, prediction и target требуют qa или turn_with_history, если документы явно
не задают отдельную агрегацию до dialogue score.
Для одного готового score используй identity. Majority используй, только
если документы задают решение большинством. all_assessors используй, только если
документы требуют минимум/единогласие: результат равен 1, только если каждый
assessor_vote равен 1. Оба метода содержат только assessor_vote; criteria-методы только
criterion sources, accuracy ровно prediction и target. Для неприменимого criterion используй
missing_policy=exclude_value; exclude_unit означает исключение всей единицы наблюдения.
Если документы явно называют агрегацию взвешенной и layout содержит weight,
используй frequency_weighted_mean; само наличие weight не доказывает взвешивание.
Если документы задают единицу наблюдения, физический положительный count/frequency
показывает число таких наблюдений в строке и repair сообщает единственный точно
согласующийся weighted identity-план, это совместное доказательство
frequency_weighted_mean; процитируй единицу и заявленное значение. Во всех иных
случаях правило неприменимо. Без доказанного взвешивания используй mean. Если
validation report публикует macro и weighted одновременно, выбери один вариант
только по отдельному текстовому указанию, какой из них использован как целевая КМ.
Процитируй это указание и строку выбранного варианта в evidence.reducer, не цитируй
конкурирующий вариант. Без такого указания верни reported_value_state=ambiguous.
Если repair-ошибка содержит required_reducer или required_reported_value, исправь
соответствующие поля и evidence ровно по указанным spans."""


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
