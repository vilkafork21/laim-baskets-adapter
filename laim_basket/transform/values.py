"""Общие скалярные нормализаторы без политики."""

import math
import re


def blank(value: object) -> bool:
    """Пустая ячейка транспорта: None, NaN или строка из пробелов."""
    return value is None or (
        isinstance(value, float) and math.isnan(value)
    ) or str(value).strip() == ""

_SPACES = re.compile(r"[\s  ]+")
_SLUG_JUNK = re.compile(r"[^a-zа-яё0-9]+")


def slug(name: str) -> str:
    text = _SLUG_JUNK.sub("_", name.casefold().replace("ё", "е")).strip("_")
    return text or "metric"


def normalize_key(value: object) -> str:
    return _SPACES.sub(" ", str(value)).strip().casefold().replace("ё", "е")


def coerce_numeric(value: object) -> float:
    if value is None or str(value).strip() == "":
        return math.nan
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric score")
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        text = _SPACES.sub("", str(value))
        percent = text.endswith("%")
        result = float(text.rstrip("%").replace(",", "."))
        if percent:
            result /= 100
    if not math.isfinite(result):
        raise ValueError("numeric score must be finite")
    return result
