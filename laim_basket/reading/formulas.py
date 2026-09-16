"""Формулы xlsx: ЕДИНСТВЕННЫЙ предикат «вертикального агрегата».

Вход: карта формул листа {(row, col) 0-based → текст формулы}.
Обработка: вертикальный агрегат подводит итог КОЛОНКИ — SUM/COUNT/… по
диапазону, пересекающему несколько строк (`=COUNT(U2:U364)`,
`=СУММ($C$2:$C$364)`, `=SUM(C:C)`); построчные агрегаты по ячейкам своей
строки (`=IF(SUM(E269:G269)/3=1,…)`) — это данные, не футер.
Выход: множество ячеек с вертикальными агрегатами.
Потребители: автотрим резолвера, _prove_tail, footer_candidates evidence —
один предикат вместо трёх рассинхронизированных.
"""

import re

from .. import defaults

_AGGREGATE = re.compile(defaults.AGGREGATE_FORMULA_PATTERN, re.IGNORECASE)
# A2:B364, $C$2:$C$364, C$2:C364 — диапазон с номерами строк
_BOUNDED_RANGE = re.compile(r"\$?[A-Z]{1,3}\$?(\d+):\$?[A-Z]{1,3}\$?(\d+)")
# C:C, $C:$C — целые колонки (всегда вертикальны); границы, чтобы не матчить середину A1:B2
_WHOLE_COLUMN = re.compile(r"(?<![A-Z0-9$:])\$?[A-Z]{1,3}:\$?[A-Z]{1,3}(?![A-Z0-9:])")


def is_vertical_aggregate(formula: str) -> bool:
    if not _AGGREGATE.search(formula):
        return False
    if _WHOLE_COLUMN.search(formula):
        return True
    return any(int(r1) != int(r2) for r1, r2 in _BOUNDED_RANGE.findall(formula))


def vertical_aggregate_cells(formulas: dict[tuple[int, int], str]) -> set[tuple[int, int]]:
    """Ячейки с вертикальными агрегатами — один O(F)-проход, дальше lookup."""
    return {cell for cell, formula in formulas.items() if is_vertical_aggregate(formula)}


# Разрешаем только статические A1-ссылки текущей строки. Текстовые литералы
# не являются ссылками; INDIRECT/OFFSET и внешние источники доказать нельзя.
_LOCAL_RANGE = re.compile(r"\$?[A-Z]{1,3}\$?([1-9][0-9]*)(?::\$?[A-Z]{1,3}\$?([1-9][0-9]*))?", re.I)
_DYNAMIC_FUNCTIONS = {
    "INDIRECT",
    "OFFSET",
    "RAND",
    "RANDBETWEEN",
    "RANDARRAY",
    "NOW",
    "TODAY",
    "WEBSERVICE",
    "RTD",
    "ДВССЫЛ",
    "СМЕЩ",
    "СЛЧИС",
    "СЛУЧМЕЖДУ",
    "ТДАТА",
    "СЕГОДНЯ",
}


def is_row_local_formula(formula: str, row: int) -> bool:
    """row — физический номер строки Excel (1-based); формулы не исполняются."""
    from openpyxl.formula.tokenizer import Tokenizer, TokenizerError

    if not isinstance(formula, str) or not formula.startswith("="):
        return False
    try:
        tokens = Tokenizer(formula).items
    except (TokenizerError, IndexError, ValueError):
        return False
    if not tokens:
        return False
    for token in tokens:
        if token.type == "FUNC" and token.subtype == "OPEN":
            if token.value[:-1].upper().split(".")[-1] in _DYNAMIC_FUNCTIONS:
                return False
        if token.type == "OPERAND" and token.subtype == "RANGE":
            ref = _LOCAL_RANGE.fullmatch(token.value)
            if ref is None or any(int(value) != row for value in ref.groups() if value is not None):
                return False
    return True
