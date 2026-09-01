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
