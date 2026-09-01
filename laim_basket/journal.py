"""Журнал прогона: каждое событие немедленно в logging и в буфер для km_result."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

CONTRACT_VERSION = "laim-run-report.v1"


class Journal:
    def __init__(self) -> None:
        self.stages: list[dict] = []
        self.decisions: dict[str, object] = {}
        self.warnings: list[dict] = []
        self.dropped_rows: dict[str, list[int]] = {}
        self.input_sha256: dict[str, str] = {}
        self.llm: dict[str, object] = {}

    def stage(self, name: str, outcome: str, ms: int) -> None:
        self.stages.append({"stage": name, "ms": ms, "outcome": outcome})
        logger.info("Этап %s: %s за %d мс", name, outcome, ms)

    def decision(self, **fields) -> None:
        self.decisions.update(fields)
        logger.info("Решение: %s",
                    ", ".join(f"{key}={value!r}" for key, value in fields.items()))

    def warning(self, code: str, message: str) -> None:
        self.warnings.append({"code": code, "message": message})
        logger.warning("%s: %s", code, message)

    def dropped(self, kind: str, rows: list[int]) -> None:
        self.dropped_rows[kind] = list(rows)
        logger.warning("Отброшены строки (%s): %s", kind, rows)

    def set_inputs(self, hashes: dict[str, str]) -> None:
        """Хэши входов уже посчитаны при скане пакета — не читать файлы повторно."""
        self.input_sha256 = dict(hashes)

    def set_llm(self, *, model: str, structured_output: bool | None,
                calls: int, repair_turns: int, transport_retries: int) -> None:
        self.llm = {"model": model, "structured_output": structured_output,
                    "calls": calls, "repair_turns": repair_turns,
                    "transport_retries": transport_retries}

    def report(self, *, basket_id: str, status: str, km: dict | None) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "basket_id": basket_id,
            "status": status,
            "km": km,
            "decisions": dict(self.decisions),
            "stages": list(self.stages),
            "llm": dict(self.llm),
            "warnings": list(self.warnings),
            "dropped_rows": {key: list(value) for key, value in self.dropped_rows.items()},
            "input_sha256": dict(self.input_sha256),
        }
