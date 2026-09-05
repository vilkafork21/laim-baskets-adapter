"""JSON-схемы двух LLM-задач.

Поле quotes стоит в properties первым: при последовательной генерации модель
сперва выписывает основание из документов, затем решает. Физические границы
данных и режим оценки в схемах отсутствуют — ими владеет только код.
"""
from __future__ import annotations

_ADDRESS = {"type": "string", "pattern": "^[A-Z]{1,3}$"}
_NULLABLE_ADDRESS = {"oneOf": [{"$ref": "#/$defs/address"}, {"type": "null"}]}

_QUOTES = {
    "type": "object",
    "additionalProperties": {"type": "array", "items": {"type": "string"}},
}

_OUTPUT_ADDRESS = {
    "oneOf": [
        {"$ref": "#/$defs/address"},
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["coalesce"],
            "properties": {
                "coalesce": {
                    "type": "array",
                    "minItems": 2,
                    "items": {"$ref": "#/$defs/address"},
                },
            },
        },
        {"type": "null"},
    ],
}

LAYOUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$defs": {"address": _ADDRESS},
    "type": "object",
    "additionalProperties": False,
    "required": ["quotes", "sheet_name", "header_rows", "roles", "grouping",
                 "dialogue_blob", "weight_column"],
    "properties": {
        "quotes": _QUOTES,
        "sheet_name": {"type": "string", "minLength": 1},
        "header_rows": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"type": "integer", "minimum": 1},
        },
        "roles": {
            "type": "object",
            "additionalProperties": False,
            "required": ["query_id", "session_id", "input_query", "output_answer",
                          "scenario", "assessor_id", "reference_answers"],
            "properties": {
                "query_id": _NULLABLE_ADDRESS,
                "session_id": _NULLABLE_ADDRESS,
                "input_query": {"$ref": "#/$defs/address"},
                "output_answer": _OUTPUT_ADDRESS,
                "scenario": _NULLABLE_ADDRESS,
                "assessor_id": _NULLABLE_ADDRESS,
                "reference_answers": {"type": "array",
                                       "items": {"$ref": "#/$defs/address"}},
            },
        },
        "grouping": {
            "type": "object", "additionalProperties": False,
            "required": ["kind", "column"],
            "properties": {
                "kind": {"enum": ["none", "merged_rows", "column", "blob_row"]},
                "column": _NULLABLE_ADDRESS,
            },
        },
        "dialogue_blob": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object", "additionalProperties": False,
                    "required": ["column", "container", "question_marker",
                                  "answer_marker"],
                    "properties": {
                        "column": {"$ref": "#/$defs/address"},
                        "container": {"enum": ["python_list", "plain_text"]},
                        "question_marker": {"type": "string", "minLength": 1},
                        "answer_marker": {"type": "string", "minLength": 1},
                    },
                },
            ],
        },
        "weight_column": _NULLABLE_ADDRESS,
    },
}

_SOURCE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["column_id", "role", "normalization", "polarity"],
    "properties": {
        "column_id": _ADDRESS,
        "role": {"enum": ["final_score", "criterion", "assessor_vote",
                           "prediction", "target"]},
        "normalization": {
            "oneOf": [
                {"enum": ["numeric", "percent", "label"]},
                {"type": "object", "minProperties": 1,
                 "additionalProperties": {"type": "number"}},
            ],
        },
        "polarity": {"enum": ["direct", "inverted"]},
    },
}

METRIC_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["quotes", "metric_name", "method", "sources", "reducer",
                 "missing_policy", "majority_denominator", "scale",
                 "reported_value", "threshold", "comparator", "assessment_mode", "evaluation"],
    "properties": {
        "quotes": _QUOTES,
        "assessment_mode": {"enum": ["qa", "turn_with_history", "dialogue"]},
        "evaluation": {
            "type": "object", "additionalProperties": False,
            "required": ["rubric", "score_values", "higher_is_better", "defect_threshold",
                         "required_evidence", "prediction_observable", "observation_profile"],
            "properties": {
                "rubric": {"type": "string", "minLength": 1},
                "score_values": {"type": "array", "minItems": 2, "uniqueItems": True,
                                 "items": {"type": "number"}},
                "higher_is_better": {"type": "boolean"},
                "defect_threshold": {"type": "number"},
                "required_evidence": {"type": "array", "uniqueItems": True,
                    "items": {"enum": ["history", "knowledge_context", "tool_results", "customer_context"]}},
                "external_party": {"type": ["string", "null"]},
                "prediction_observable": {"enum": ["route_label", "output_answer", None]},
                "observation_profile": {"enum": ["fipa_external_reply_v1", "aef_boundary_v1", "state_single_request_v1"]},
            },
        },
        "metric_name": {"type": "string", "minLength": 1},
        "method": {"enum": ["identity", "accuracy", "mean_criteria",
                             "all_criteria", "all_assessors", "majority"]},
        "sources": {"type": "array", "minItems": 1, "items": _SOURCE},
        "reducer": {"enum": ["mean", "frequency_weighted_mean"]},
        "missing_policy": {"enum": ["fail", "exclude_unit", "exclude_value", "zero"]},
        "majority_denominator": {"enum": ["declared", "present", None]},
        "scale": {"enum": ["ratio", "percent", "raw"]},
        "reported_value": {
            "type": "object", "additionalProperties": False,
            "required": ["state", "value", "raw"],
            "properties": {
                "state": {"enum": ["declared", "not_declared", "ambiguous"]},
                "value": {"type": ["number", "null"]},
                # Точный текст числа из отчёта: от него считаются precision
                # и допуск сверки (последний опубликованный разряд).
                "raw": {"type": ["string", "null"]},
            },
        },
        "threshold": {"type": ["number", "null"]},
        "comparator": {"enum": [">=", "<=", None]},
    },
}
