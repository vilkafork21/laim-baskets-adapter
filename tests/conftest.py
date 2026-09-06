from __future__ import annotations

import sys
from pathlib import Path

NODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NODE_ROOT))


def source(column_id: str, role: str, normalization: object = "numeric",
           polarity: str = "direct") -> dict:
    return {
        "column_id": column_id, "role": role,
        "normalization": normalization, "polarity": polarity,
    }


def layout_answer(**overrides) -> dict:
    """Валидный ответ LLM на задачу разметки: плоская Q/A-корзина A/B/C."""
    answer = {
        "quotes": {},
        "sheet_name": "Лист1",
        "header_rows": [1],
        "roles": {"query_id": None, "session_id": None, "input_query": "A",
                   "output_answer": "B", "scenario": None,
                   "assessor_id": None, "reference_answers": []},
        "grouping": {"kind": "none", "column": None},
        "dialogue_blob": None,
        "weight_column": None,
    }
    answer.update(overrides)
    return answer


def evaluation_answer(**overrides) -> dict:
    result = {
        "rubric": "Оценить правильность ответа: 1 — верно, 0 — неверно",
        "score_values": [0, 1], "higher_is_better": True, "defect_threshold": 1,
        "required_evidence": [], "prediction_observable": None,
        "observation_profile": "state_single_request_v1",
    }
    result.update(overrides)
    return result


def metric_answer(**overrides) -> dict:
    """Валидный ответ LLM на задачу метрики: identity по колонке C."""
    answer = {
        "quotes": {"metric": ["Accuracy 0.5"]},
        "metric_name": "Accuracy",
        "assessment_mode": "qa",
        "evaluation": evaluation_answer(),
        "method": "identity",
        "sources": [source("C", "final_score")],
        "reducer": "mean",
        "missing_policy": "exclude_unit",
        "majority_denominator": None,
        "scale": "ratio",
        "reported_value": {"state": "declared", "value": 0.5, "raw": "0.5"},
        "threshold": None,
        "comparator": None,
    }
    answer.update(overrides)
    return answer


def layout_proposal(roles: dict, grouping=None, dialogue_blob=None, weight=None,
                    *, sheet_name: str = "Sheet",
                    header_rows: tuple[int, ...] = (1,)) -> dict:
    if isinstance(dialogue_blob, dict):
        dialogue_blob = dict(dialogue_blob)
    return {
        "quotes": {},
        "sheet_name": sheet_name,
        "header_rows": list(header_rows),
        "roles": {
            "query_id": None, "session_id": None, "scenario": None,
            "assessor_id": None, "reference_answers": [], **roles,
        },
        "grouping": grouping or {"kind": "none", "column": None},
        "dialogue_blob": dialogue_blob,
        "weight_column": weight,
    }
