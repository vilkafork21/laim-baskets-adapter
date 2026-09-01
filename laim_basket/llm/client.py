"""OpenAI-совместимый клиент контурного LLM-шлюза.

Reasoning-модели шлюза кладут рассуждения в <think>…</think> внутри content
(вырезаем; незакрытый think = обрыв ответа, retryable). Каждый вызов
персистится (запрос без Authorization + сырой ответ + извлечённый контент) —
это аудит прогона.
"""

import json
import logging
import re
import time
from pathlib import Path

import jsonschema
import requests

from .. import defaults
from ..config import LlmConfig
from ..errors import BasketError, LlmError, StructuredOutputError

logger = logging.getLogger(__name__)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCED = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_TRANSPORT_RETRIES = 3

_REPAIR = (
    "Твой предыдущий ответ отклонён валидатором. Точная причина:\n{error}\n"
    "Верни ИСПРАВЛЕННЫЙ полный JSON-объект (не диф) и ничего кроме JSON."
)


def _content_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):  # некоторые шлюзы отдают список частей
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str):
        raise LlmError("В ответе нет текстового content", message_keys=sorted(message))
    if "<think>" in content and "</think>" not in content:
        # контурный формат обрыва бюджета (reasoning внутри content) —
        # обязан эскалировать max_tokens так же, как finish_reason=length
        raise LlmError("Рассуждения оборваны (незакрытый <think>) — усечённый ответ",
                       retryable=True, truncated=True)
    return _THINK.sub("", content).strip()


class LlmClient:
    def __init__(self, config: LlmConfig, out_dir: Path | None = None):
        self.config = config
        self.out_dir = out_dir
        self._calls = 0
        # выученный бюджет токенов живёт на клиенте ВЕСЬ прогон: repair-ходы
        # и паспорт не переплачивают за повторное обучение эскалации
        self._max_tokens = config.max_tokens
        self._tokens_ceiling = max(defaults.MAX_TOKENS_CEILING, config.max_tokens)
        # Поддержка response_format шлюзом неизвестна заранее: проба на первом
        # вызове, итог запоминается на весь прогон (None -> True/False).
        self.structured_output: bool | None = None
        self.calls = 0
        self.repair_turns = 0
        self.transport_retries = 0

    def _persist(self, label: str, body: dict, status: int, payload) -> None:
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._calls += 1
        record = {
            "url": self.config.url, "model": self.config.model,
            "request": body, "http_status": status, "response": payload,
        }
        path = self.out_dir / f"{label}_call_{self._calls}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def _attempt(self, body: dict, headers: dict, label: str) -> str:
        """Одна попытка. LlmError с retryable=True — кандидат на повтор."""
        try:
            response = requests.post(
                self.config.url, headers=headers, json=body,
                timeout=(self.config.connect_timeout, self.config.read_timeout),
            )
        except requests.RequestException as exc:
            raise LlmError(f"Транспортная ошибка: {exc}", retryable=True) from exc

        try:
            payload = response.json()
        except ValueError:
            payload = {"raw_text": response.text[:defaults.ERROR_PAYLOAD_CHAR_CAP]}
        self._persist(label, body, response.status_code, payload)

        if response.status_code in _RETRYABLE_STATUS:
            raise LlmError(f"HTTP {response.status_code} от шлюза", retryable=True)
        if response.status_code != 200:
            raise LlmError(f"HTTP {response.status_code}: {str(payload)[:300]}",
                           status=response.status_code)
        try:
            choice = payload["choices"][0]
            if choice.get("finish_reason") == "length" and not choice["message"].get("content"):
                raise LlmError(
                    "Ответ обрезан по max_tokens (рассуждения съели бюджет) — повтор",
                    retryable=True, truncated=True,
                )
            return _content_text(choice["message"])
        except (KeyError, IndexError) as exc:
            raise LlmError(f"Неожиданная структура ответа: {exc}",
                           payload_keys=sorted(payload), retryable=True) from exc

    def chat(self, messages: list[dict], label: str,
             response_schema: dict | None = None) -> str:
        self.calls += 1
        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        if response_schema is not None and self.structured_output is not False:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": label, "schema": response_schema},
            }
        headers = {"Authorization": f"Bearer {self.config.api_key}",
                   "Content-Type": "application/json"}

        last_error = None
        attempt = 0
        while attempt < _TRANSPORT_RETRIES:
            try:
                content = self._attempt(body, headers, label)
                if "response_format" in body and self.structured_output is None:
                    self.structured_output = True
                return content
            except LlmError as exc:
                if exc.details.get("status") == 400 and "response_format" in body:
                    # шлюз не принял structured output — фолбэк на весь прогон;
                    # попытка не тратится: это не транспортная ошибка
                    body.pop("response_format")
                    self.structured_output = False
                    logger.warning(
                        "Шлюз отклонил response_format (HTTP 400) — "
                        "фолбэк на текстовый JSON до конца прогона")
                    continue
                if not exc.details.get("retryable"):
                    raise
                if exc.details.get("truncated"):
                    # адаптивная эскалация: рассуждения не влезли — удваиваем
                    # бюджет до потолка; выученное значение персистится на клиенте
                    self._max_tokens = min(self._max_tokens * 2, self._tokens_ceiling)
                    body["max_tokens"] = self._max_tokens
                last_error = exc
                attempt += 1
                self.transport_retries += 1
                time.sleep(2 * attempt)
        raise last_error or LlmError("LLM недоступна после ретраев")


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


def request_structured(client: LlmClient, messages: list[dict], schema: dict,
                       label: str, validate_extra=None, max_turns: int = 3) -> dict:
    """Цикл structured-JSON: schema в response_format (если шлюз умеет) и в
    промпте; локальная валидация всегда; validate_extra(obj) может бросить
    BasketError — её текст уходит модели как repair."""
    history = list(messages)
    last_error, last_exception = None, None
    for turn in range(1, max_turns + 1):
        content = client.chat(history, f"{label}_turn{turn}", response_schema=schema)
        try:
            obj = extract_json(content)
            jsonschema.validate(obj, schema)
            if validate_extra is not None:
                validate_extra(obj)
            return obj
        except jsonschema.ValidationError as exc:
            last_error = f"JSON Schema: {exc.message} (path: {exc.json_path})"
            last_exception = exc
        except LlmError as exc:
            last_error, last_exception = str(exc), exc
        except Exception as exc:  # BasketError из validate_extra
            last_error, last_exception = f"{type(exc).__name__}: {exc}", exc
            details = getattr(exc, "details", None)
            if details:  # без деталей (known_columns/missing) модель слепа
                last_error += " | детали: " + json.dumps(
                    details, ensure_ascii=False, default=str
                )[:defaults.ERROR_PAYLOAD_CHAR_CAP]
        # Без этой записи причина repair видна только внутри следующего промпта:
        # прогон выглядит как немотивированное «модель не смогла».
        logger.warning("%s: попытка %d/%d отклонена — %s",
                       label, turn, max_turns, last_error)
        client.repair_turns += 1
        history.append({"role": "assistant", "content": content})
        history.append({"role": "user", "content": _REPAIR.format(error=last_error)})
    if isinstance(last_exception, BasketError) and not isinstance(last_exception, LlmError):
        raise last_exception
    raise StructuredOutputError(
        f"LLM не дала валидный {label} за {max_turns} попыток; "
        f"последняя ошибка: {last_error}",
        label=label,
    )
