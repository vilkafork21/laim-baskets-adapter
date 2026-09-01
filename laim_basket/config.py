"""Конфигурация запуска и LLM-пресеты.

Паритет сред — ключевой принцип: между локальным (openrouter) и контурным
(contour) пресетами различаются ТОЛЬКО base_url, имя модели и источник ключа.
Параметры сэмплирования, капы и промпты общие и здесь не переопределяются.
"""

import os
from dataclasses import dataclass, field

from .errors import LlmError

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CONTOUR_URL = "http://sds-ai-gateway:8097/api/v1/chat/completions"


@dataclass(frozen=True)
class LlmConfig:
    preset: str
    url: str
    model: str
    api_key: str
    # Общие для всех пресетов параметры (байт-в-байт одинаковы в обеих средах).
    temperature: float = 0.0
    top_p: float = 0.05
    # Reasoning-модели тратят бюджет на <think>: запас обязателен, иначе
    # finish_reason=length срезает сам JSON (наблюдалось на minimax-m2.7).
    max_tokens: int = 16384
    connect_timeout: float = 10.0
    read_timeout: float = 300.0
    # Опции запроса, специфичные для провайдера; в контур этот payload не уходит.
    extra_body: dict = field(default_factory=dict)


def llm_config(preset: str) -> LlmConfig:
    """Собрать конфиг LLM по имени пресета с env-переопределениями.

    env: LAIM_LLM_URL / LAIM_LLM_MODEL перекрывают url/model любого пресета;
    ключи — OPENROUTER_API_KEY (openrouter) и AI_GATEWAY_API_KEY (contour).
    """
    if preset == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise LlmError(
                "Не задан OPENROUTER_API_KEY — экспортируйте ключ в окружение",
                preset=preset,
            )
        url, model = OPENROUTER_URL, "minimax/minimax-m3"
        extra = {"reasoning": {"enabled": False}}
    elif preset == "contour":
        # Внутренний sds-ai-gateway аутентификацию не проверяет — заглушка «123»
        # унаследована из проверенной интеграции. Для любого ДРУГОГО хоста
        # (LAIM_LLM_URL переопределён) ключ обязателен — проверка ниже.
        api_key = os.environ.get("AI_GATEWAY_API_KEY", "123")
        url, model = CONTOUR_URL, "minimax-m2.5"
        extra = {}
    else:
        raise LlmError(f"Неизвестный LLM-пресет: {preset!r}", preset=preset)

    resolved_url = os.environ.get("LAIM_LLM_URL", url)
    if preset == "contour" and resolved_url != CONTOUR_URL and "AI_GATEWAY_API_KEY" not in os.environ:
        raise LlmError(
            "LAIM_LLM_URL переопределён на сторонний шлюз — задайте AI_GATEWAY_API_KEY явно",
            preset=preset, url=resolved_url,
        )

    return LlmConfig(
        preset=preset,
        url=resolved_url,
        model=os.environ.get("LAIM_LLM_MODEL", model),
        api_key=api_key,
        extra_body=extra,
    )
