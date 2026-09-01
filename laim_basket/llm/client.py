"""OpenAI-совместимый LLM-клиент: один код для OpenRouter и контурного шлюза.

Различия сред — только url/model/key из LlmConfig. Ответ MiniMax может нести
рассуждения: OpenRouter кладёт их в reasoning_details (игнорируем), шлюз —
в <think>…</think> внутри content (вырезаем; незакрытый think = обрыв,
retryable). Каждый вызов персистится (запрос без Authorization + сырой ответ
+ извлечённый контент) — это аудит и материал для диффа сред.
"""

import json
import re
import time
from pathlib import Path

import requests

from .. import defaults
from ..config import LlmConfig
from ..errors import LlmError

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_TRANSPORT_RETRIES = 3


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

    def _persist(self, label: str, body: dict, status: int, payload) -> None:
        if self.out_dir is None:
            return
        self._calls += 1
        record = {
            "url": self.config.url, "model": self.config.model, "preset": self.config.preset,
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

    def chat(self, messages: list[dict], label: str) -> str:
        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self._max_tokens,
            "stream": False,
            **self.config.extra_body,
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}",
                   "Content-Type": "application/json"}

        last_error = None
        for attempt in range(1, _TRANSPORT_RETRIES + 1):
            try:
                return self._attempt(body, headers, label)
            except LlmError as exc:
                if not exc.details.get("retryable"):
                    raise
                if exc.details.get("truncated"):
                    # адаптивная эскалация: рассуждения не влезли — удваиваем
                    # бюджет до потолка; выученное значение персистится на клиенте
                    self._max_tokens = min(self._max_tokens * 2, self._tokens_ceiling)
                    body["max_tokens"] = self._max_tokens
                last_error = exc
                time.sleep(2 * attempt)
        raise last_error or LlmError("LLM недоступна после ретраев")
