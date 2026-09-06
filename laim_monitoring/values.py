"""Скалярные помощники: пустое значение, ключ сравнения, Decimal."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import pandas as pd

from .errors import MonitoringContractError


def blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def to_decimal(value: object, name: str = "value") -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise MonitoringContractError(f"{name} не является Decimal: {value!r}") from exc
    if not result.is_finite():
        raise MonitoringContractError(f"{name} должно быть конечным")
    return result


def value_key(value: object) -> tuple[str, str]:
    """Ключ равенства значений разных транспортов: 1, 1.0 и "1" — одно и то же."""
    if isinstance(value, bool):
        return "bool", str(value)
    if isinstance(value, (int, float, Decimal)) and not blank(value):
        return "number", str(to_decimal(value).normalize())
    return "text", str(value).strip().casefold()
