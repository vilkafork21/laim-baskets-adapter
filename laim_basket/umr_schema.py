"""Минимальная валидация УМР как гейт публикации."""

import math
import re
from numbers import Integral

from .transform.values import blank

REQUIRED_COLUMNS = ("query_id", "input_query", "output_answer")
_METRIC = re.compile(r"^(?:\S+_metric|main_metric)$")


def validate_flat_canon(
    frame, canon_columns: set[str] | None = None, *, session_scoped: bool = False
) -> dict[str, object]:
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    missing_values = {}
    for name in ("query_id", "input_query"):
        if name in frame.columns:
            positions = [
                index for index, value in enumerate(frame[name].tolist())
                if blank(value)
            ]
            if positions:
                missing_values[name] = {
                    "count": len(positions),
                    "row_positions": positions,
                    "source_rows": (
                        frame["source_row_id"].iloc[positions[:20]].tolist()
                        if "source_row_id" in frame.columns
                        else positions[:20]
                    ),
                }
    type_violations = []
    for name in frame.columns:
        if canon_columns is not None and name not in canon_columns:
            continue
        if _METRIC.match(str(name)):
            bad = sum(
                not blank(value)
                and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)))
                for value in frame[name].tolist()
            )
            if bad:
                type_violations.append({"column": str(name), "bad_values": bad})
    context_violations = []
    if "query_id" in frame.columns:
        present_ids = [
            value for value in frame["query_id"].tolist()
            if not blank(value)
        ]
        if session_scoped:
            identity_keys = [
                (
                    type(frame["session_id"].iloc[position]).__name__,
                    str(frame["session_id"].iloc[position]),
                    type(value).__name__,
                    str(value),
                )
                for position, value in enumerate(frame["query_id"].tolist())
                if not blank(value)
            ]
        else:
            identity_keys = [(type(value).__name__, str(value)) for value in present_ids]
        if len(identity_keys) != len(set(identity_keys)):
            context_violations.append({
                "reason": (
                    "duplicate_session_query_id" if session_scoped else "duplicate_query_id"
                )
            })
    has_group = "reference_group_id" in frame
    has_order = "turn_index" in frame
    if has_group != has_order:
        context_violations.append({
            "reason": "paired_columns_required",
            "missing": "turn_index" if has_group else "reference_group_id",
        })
    elif has_group:
        groups: dict[str, list[int]] = {}
        blank_groups = []
        invalid_indexes = []
        for position, (group, index) in enumerate(
            zip(frame["reference_group_id"].tolist(), frame["turn_index"].tolist())
        ):
            if blank(group):
                blank_groups.append(position)
                continue
            if isinstance(index, bool) or not isinstance(index, Integral) or index < 1:
                invalid_indexes.append(position)
                continue
            groups.setdefault(str(group), []).append(int(index))
        if blank_groups:
            context_violations.append({"reason": "blank_group", "row_positions": blank_groups})
        if invalid_indexes:
            context_violations.append({
                "reason": "invalid_turn_index",
                "row_positions": invalid_indexes,
            })
        for group, indexes in groups.items():
            if len(indexes) != len(set(indexes)):
                context_violations.append({"reason": "duplicate_turn_index", "group": group})
            if 1 not in indexes:
                context_violations.append({"reason": "turn_index_must_start_at_1", "group": group})
    status = (
        "passed"
        if not missing_columns and not missing_values and not type_violations and not context_violations
        else "failed"
    )
    return {
        "status": status,
        "missing_required": missing_columns,
        "missing_required_values": missing_values,
        "type_violations": type_violations,
        "context_violations": context_violations,
    }
