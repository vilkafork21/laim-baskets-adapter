"""Имена колонок из строк шапки — ЕДИНСТВЕННАЯ реализация.

Использовать обязаны и evidence (что видит LLM), и регион (что требует
резолвер): расхождение этих двух доменов имён — доказанный сбой (CI09840670:
модель не могла угадать имена региона по сырым заголовкам evidence).
Правила: merged-заголовок раздаётся всем колонкам диапазона, части строк
склеиваются пробелом, пробелы/переводы строк схлопываются, пустые имена —
`unnamed_<буква>`, дубли — `имя#2`, `имя#3`…
"""

import re

from openpyxl.utils import get_column_letter

from .xlsx_reader import RawSheet

_SPACES = re.compile(r"\s+")


def header_merge_map(sheet: RawSheet, header_rows0: list[int]) -> dict[tuple[int, int], tuple[int, int]]:
    """(r, c) шапки внутри merged-диапазона → его левая верхняя ячейка.

    Берётся любой merge, ПЕРЕСЕКАЮЩИЙ строки шапки (в т.ч. начавшийся выше —
    баннер А1:А3 при шапке [2,3]); экспансия ограничена строками шапки, чтобы
    вертикальный merge через данные не раздувал карту."""
    lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for r1, c1, r2, c2 in sheet.merged:
        rows_hit = [r for r in header_rows0 if r1 <= r <= r2]
        for r in rows_hit:
            for c in range(c1, c2 + 1):
                lookup[(r, c)] = (r1, c1)
    return lookup


def merge_header(sheet: RawSheet, header_rows0: list[int], c: int,
                 merge_map: dict[tuple[int, int], tuple[int, int]]) -> str:
    """Имя колонки: непустые части строк шапки сверху вниз через пробел.

    Одна merged-ячейка, пересекающая обе строки шапки, даёт свою часть один
    раз (иначе вертикальный merge «id» A1:A2 дал бы «id id»)."""
    parts, seen_anchors = [], set()
    for r in header_rows0:
        anchor = merge_map.get((r, c), (r, c))
        if anchor in seen_anchors:
            continue
        seen_anchors.add(anchor)
        rr, cc = anchor
        value = sheet.grid[rr][cc] if rr < sheet.n_rows else None
        text = _SPACES.sub(" ", str(value)).strip() if value is not None else ""
        if text:
            parts.append(text)
    return " ".join(parts)


def dedup_names(names: list[str]) -> tuple[list[str], dict[str, str]]:
    """Дубли имён → «имя#2», «имя#3»… (карта переименований — в отчёт)."""
    seen: dict[str, int] = {}
    result, renames = [], {}
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        if seen[name] == 1:
            result.append(name)
        else:
            new_name = f"{name}#{seen[name]}"
            result.append(new_name)
            renames[new_name] = name
    return result, renames


def header_names(sheet: RawSheet, header_rows: list[int]) -> list[str]:
    """Итоговые имена всех колонок листа для заданных 1-based строк шапки —
    ровно те, которые примет резолвер при table.header_rows == header_rows."""
    header_rows0 = [r - 1 for r in header_rows]
    merge_map = header_merge_map(sheet, header_rows0)
    raw = [merge_header(sheet, header_rows0, c, merge_map) for c in range(sheet.n_cols)]
    named = [name or f"unnamed_{get_column_letter(c + 1)}" for c, name in enumerate(raw)]
    deduped, _renames = dedup_names(named)
    return deduped
