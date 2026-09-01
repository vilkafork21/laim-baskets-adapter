"""Конфигурация обращения к контурному LLM-шлюзу.

Сэмплирование детерминированное и здесь не настраивается: модель задаётся
настройкой ноды `model_id`, адрес шлюза — переменной окружения только для
стендов.
"""

import os
from dataclasses import dataclass

from .errors import LlmError

CONTOUR_URL = "http://sds-ai-gateway:8097/api/v1/chat/completions"


@dataclass(frozen=True)
class LlmConfig:
    url: str
    model: str
    api_key: str
    temperature: float = 0.0
    top_p: float = 0.05
    # Reasoning-модели тратят бюджет на <think>: запас обязателен, иначе
    # finish_reason=length срезает сам JSON (наблюдалось на minimax-m2.7).
    max_tokens: int = 16384
    connect_timeout: float = 10.0
    read_timeout: float = 300.0


def llm_config() -> LlmConfig:
    """Конфиг контурного шлюза; LAIM_LLM_URL/LAIM_LLM_MODEL — для стендов."""
    # Внутренний sds-ai-gateway аутентификацию не проверяет — заглушка «123»
    # унаследована из проверенной интеграции. Для любого ДРУГОГО хоста
    # (LAIM_LLM_URL переопределён) ключ обязателен.
    api_key = os.environ.get("AI_GATEWAY_API_KEY", "123")
    url = os.environ.get("LAIM_LLM_URL", CONTOUR_URL)
    if url != CONTOUR_URL and "AI_GATEWAY_API_KEY" not in os.environ:
        raise LlmError(
            "LAIM_LLM_URL переопределён на сторонний шлюз — задайте AI_GATEWAY_API_KEY явно",
            url=url,
        )
    return LlmConfig(
        url=url,
        model=os.environ.get("LAIM_LLM_MODEL", "glm-5.2"),
        api_key=api_key,
    )
