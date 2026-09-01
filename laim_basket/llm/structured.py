"""Цикл structured-JSON: извлечение → схема → repair-ретрай с текстом ошибки.

LLM возвращает только JSON (возможен fenced-блок или обрамляющий текст);
невалидный ответ не чинится кодом — модель получает точный текст нарушения
и исправляет сама, максимум max_turns попыток.
"""

import json
import re

import jsonschema

from .. import defaults
from ..errors import BasketError, LlmError, StructuredOutputError

_FENCED = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_REPAIR = (
    "Твой предыдущий ответ отклонён валидатором. Точная причина:\n{error}\n"
    "Верни ИСПРАВЛЕННЫЙ полный JSON-объект (не диф) и ничего кроме JSON."
)


def extract_json(content: str) -> dict:
    for match in _FENCED.finditer(content):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
    start = content.find("{")
    if start == -1:
        raise LlmError("В ответе нет JSON-объекта", content_head=content[:200])
    try:
        obj, _ = json.JSONDecoder().raw_decode(content[start:])
    except json.JSONDecodeError as exc:
        raise LlmError(f"JSON не парсится: {exc}", content_head=content[:200]) from exc
    if not isinstance(obj, dict):
        raise LlmError("Ожидался JSON-объект, получено иное")
    return obj


def request_structured(client, messages: list[dict], schema: dict, label: str,
                       validate_extra=None, max_turns: int = 3) -> dict:
    """validate_extra(obj) может бросить BasketError — её текст уходит в repair."""
    history = list(messages)
    last_error, last_exception = None, None
    for turn in range(1, max_turns + 1):
        content = client.chat(history, f"{label}_turn{turn}")
        try:
            obj = extract_json(content)
            jsonschema.validate(obj, schema)
            if validate_extra is not None:
                validate_extra(obj)
            return obj
        except jsonschema.ValidationError as exc:
            last_error, last_exception = f"JSON Schema: {exc.message} (path: {exc.json_path})", exc
        except LlmError as exc:
            last_error, last_exception = str(exc), exc
        except Exception as exc:  # BasketError из validate_extra
            last_error, last_exception = f"{type(exc).__name__}: {exc}", exc
            details = getattr(exc, "details", None)
            if details:  # known_columns/missing/unmapped — без них модель слепа
                last_error += " | детали: " + json.dumps(
                    details, ensure_ascii=False, default=str
                )[:defaults.ERROR_PAYLOAD_CHAR_CAP]
        history.append({"role": "assistant", "content": content})
        history.append({"role": "user", "content": _REPAIR.format(error=last_error)})
    if isinstance(last_exception, BasketError) and not isinstance(last_exception, LlmError):
        raise last_exception
    raise StructuredOutputError(
        f"LLM не дала валидный {label} за {max_turns} попыток; последняя ошибка: {last_error}",
        label=label,
    )
