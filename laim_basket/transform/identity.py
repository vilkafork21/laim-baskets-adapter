"""Идентичность строк канона: query_id / session_id / input_query_count."""

import math
from dataclasses import dataclass

from ..errors import LayoutError
from .values import blank, coerce_numeric


@dataclass
class IdentityResult:
    query_id: list
    session_id: list[object]
    input_query_count: list[int]
    accounting: dict


_FLOAT_ID_PRECISION_LIMIT = 2 ** 53  # выше — float уже потерял младшие разряды


def _render_id_values(values: list) -> tuple[list[str] | None, str | None]:
    """Значения колонки как id: None + причина, если колонка непригодна."""
    if any(blank(v) for v in values):
        return None, "в колонке есть пропуски"
    if any(isinstance(v, float) and abs(v) >= _FLOAT_ID_PRECISION_LIMIT for v in values):
        return None, "float с потерянной точностью (≥2^53) — id повреждён Excel"
    rendered = [
        str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).strip()
        for v in values
    ]
    return rendered, None


def _usable_query_ids(
    values: list, session_ids: list[object] | None
) -> tuple[list[str] | None, str | None]:
    rendered, reason = _render_id_values(values)
    if rendered is None:
        return None, reason
    keys = list(zip(session_ids, rendered)) if session_ids is not None else rendered
    if len(set(keys)) != len(keys):
        key_name = "(session_id, query_id)" if session_ids is not None else "query_id"
        return None, f"{key_name} повторяется"
    return rendered, None


def _weight_counts(rows: list[list], columns: list[str], source: str) -> list[int]:
    """Вес (freq) → input_query_count: обязан быть целым числом в каждой строке.

    Пустые/нечисловые/дробные значения — fail-closed: вес участвует в КМ,
    молчаливые int(NaN)/усечения недопустимы.
    """
    index = columns.index(source)
    counts, bad = [], []
    for i, row in enumerate(rows):
        try:
            value = coerce_numeric(row[index])
        except ValueError:
            bad.append({"row": i, "value": str(row[index])[:40]})
            continue
        if math.isnan(value) or value != int(value) or value < 1:
            bad.append({"row": i, "value": str(row[index])[:40]})
        else:
            counts.append(int(value))
    if bad:
        raise LayoutError(
            f"Вес {source!r} обязан быть целым числом ≥ 1 в каждой строке "
            f"(ноль/отрицательный вес ломает взвешивание и bootstrap)",
            column=source, bad_values=bad[:10], bad_count=len(bad),
        )
    return counts


def build_identity(
    rows: list[list],
    columns: list[str],
    roles: dict,
    weight: dict | None,
    groups: list[int] | None = None,
    query_id_override: list[str] | None = None,
) -> IdentityResult:
    """Идентичность строк канона.

    query_id: override группировки (blob-разворот) > пригодный источник >
    синтез row-N с причиной. session_id: явный источник > номер группы > номер
    строки. input_query_count: вес (freq) из профиля или 1.
    """
    accounting: dict = {}
    n = len(rows)

    session_role = roles.get("session_id")
    session_id = None
    if isinstance(session_role, dict):
        source = session_role["source"]
        session_id, reason = _render_id_values(
            [row[columns.index(source)] for row in rows]
        )
        if session_id is None:
            # Непригодный источник — деградация, как у query_id: корзина с
            # пропусками в session-колонке не повод ронять ноду.
            accounting["session_id"] = {
                "strategy": "derive",
                "rule": "reference_group_index" if groups is not None else "row_index",
                "source_rejected": source,
                "reason": reason,
            }
        else:
            accounting["session_id"] = {"strategy": "source", "source": source}
    if session_id is None:
        if groups is not None:
            session_id = list(groups)
        else:
            session_id = list(range(n))
        accounting.setdefault("session_id", {
            "strategy": "derive",
            "rule": "reference_group_index" if groups is not None else "row_index",
        })
    query_scope = session_id if session_role is not None or groups is not None else None

    if query_id_override is not None:
        query_id, reason = _usable_query_ids(query_id_override, query_scope)
        if query_id is None:
            query_id = [f"row-{i}" for i in range(n)]
            accounting["query_id"] = {
                "strategy": "synthesize",
                "source_rejected": "dialogue_blob",
                "reason": reason,
            }
        else:
            accounting["query_id"] = {"strategy": "unrolled", "rule": "из dialogue_blob"}
    else:
        query_id_role = roles["query_id"]
        query_id = None
        if isinstance(query_id_role, dict):
            source = query_id_role["source"]
            values = [row[columns.index(source)] for row in rows]
            query_id, reason = _usable_query_ids(values, query_scope)
            if query_id is None:
                accounting["query_id"] = {
                    "strategy": "synthesize", "source_rejected": source, "reason": reason,
                }
        if query_id is None:
            query_id = [f"row-{i}" for i in range(n)]
            accounting.setdefault(
                "query_id", {"strategy": "synthesize", "reason": "профиль: synthesize"}
            )
        else:
            accounting["query_id"] = {"strategy": "source", "source": query_id_role["source"]}

    if weight:
        counts = _weight_counts(rows, columns, weight["source"])
        accounting["input_query_count"] = {"strategy": "source", "source": weight["source"]}
    else:
        counts = [1] * n
        accounting["input_query_count"] = {"strategy": "derive", "rule": "constant_1"}

    return IdentityResult(
        query_id=query_id,
        session_id=session_id,
        input_query_count=counts,
        accounting=accounting,
    )
