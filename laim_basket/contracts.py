"""Закрытые JSON-контракты, которые возвращают два вызова LLM."""

_ADDRESS = {
    "type": "object",
    "additionalProperties": False,
    "required": ["column", "header"],
    "properties": {
        "column": {"type": "string", "pattern": "^[A-Z]{1,3}$"},
        "header": {"type": ["string", "null"]},
    },
}

_NULLABLE_ADDRESS = {"oneOf": [{"$ref": "#/$defs/address"}, {"type": "null"}]}

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
    "required": [
        "layout_version", "basket_id", "sheet_name", "ignored_sheets",
        "header_rows", "roles", "grouping", "dialogue_blob", "weight",
    ],
    "properties": {
        "layout_version": {"const": "laim-layout.v1"},
        "basket_id": {"type": "string", "minLength": 1},
        "sheet_name": {"type": "string", "minLength": 1},
        "ignored_sheets": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
        "header_rows": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"type": "integer", "minimum": 1},
        },
        # Принимается как совещательный шум модели и удаляется до резолва.
        # Владелец физических границ данных — только python.
        "first_data_row": {"type": "integer", "minimum": 1},
        "last_data_row": {"type": "integer", "minimum": 1},
        "roles": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "input_query", "output_answer", "query_id", "scenario",
                "assessor_id", "reference_answers",
            ],
            "properties": {
                "input_query": {"$ref": "#/$defs/address"},
                "output_answer": _OUTPUT_ADDRESS,
                "query_id": {
                    "oneOf": [{"$ref": "#/$defs/address"}, {"const": "synthesize"}],
                },
                "session_id": _NULLABLE_ADDRESS,
                "scenario": {
                    "oneOf": [
                        {"$ref": "#/$defs/address"},
                        {"type": "null"},
                    ],
                },
                "assessor_id": _NULLABLE_ADDRESS,
                "reference_answers": {
                    "type": "array", "items": {"$ref": "#/$defs/address"},
                },
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
                    "required": ["column", "container", "question_marker", "answer_marker"],
                    "properties": {
                        "column": {"$ref": "#/$defs/address"},
                        "container": {"enum": ["python_list", "plain_text"]},
                        "question_marker": {"type": "string", "minLength": 1},
                        "answer_marker": {"type": "string", "minLength": 1},
                    },
                },
            ],
        },
        "weight": _NULLABLE_ADDRESS,
        "evidence": {
            "type": "object",
            "additionalProperties": {"type": "string", "minLength": 1},
        },
    },
}

_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["column_id", "name", "judged"],
    "properties": {
        "column_id": {"type": "string", "pattern": "^[A-Z]{1,3}$"},
        # Имя входа в формуле.
        "name": {"type": "string", "pattern": "^[A-Za-zА-Яа-яЁё_][A-Za-zА-Яа-яЁё0-9_]{0,40}$"},
        # true — колонку проставлял разметчик (судья воспроизведёт её на мониторинге);
        # false — ответ агента, наблюдается в трейсах.
        "judged": {"type": "boolean"},
    },
}

_EVIDENCE_FIELDS = ("metric", "formula", "assessment_mode", "release", "reported_value")

MEASUREMENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "plan_version", "basket_id", "metric_name", "document_roles",
        "assessment_mode", "formula", "inputs", "release",
        "reported_value_state", "reported_value", "evidence",
    ],
    "properties": {
        "plan_version": {"const": "laim-measurement-plan.v3"},
        "basket_id": {"type": "string", "minLength": 1},
        "metric_name": {"type": "string", "minLength": 1},
        "document_roles": {
            "type": "object", "additionalProperties": False,
            "required": ["instruction", "development_report", "validation_report"],
            "properties": {
                role: {"type": "string", "pattern": "^doc-[1-3]$"}
                for role in ("instruction", "development_report", "validation_report")
            },
        },
        "assessment_mode": {"enum": ["qa", "turn_with_history", "dialogue"]},
        # Формула КМ как она определена в отчёте о валидации, над именами inputs и weight.
        "formula": {"type": "string", "minLength": 1, "maxLength": 500},
        "inputs": {"type": "array", "minItems": 1, "items": _INPUT},
        "release": {
            "type": "object", "additionalProperties": False,
            "required": ["threshold", "comparator", "scale", "precision"],
            "properties": {
                "threshold": {
                    "oneOf": [
                        {"type": "string", "pattern": "^-?[0-9]+(?:[.,][0-9]+)?$"},
                        {"type": "null"},
                    ],
                },
                "comparator": {"enum": [">=", "<=", None]},
                "scale": {"enum": ["ratio", "percent", "raw"]},
                "precision": {"type": "integer", "minimum": 0, "maximum": 12},
            },
        },
        "reported_value_state": {"enum": ["absent", "unambiguous", "ambiguous"]},
        "reported_value": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object", "additionalProperties": False,
                    "required": ["value", "raw", "span_id"],
                    "properties": {
                        "value": {"type": "string", "pattern": "^-?[0-9]+(?:[.,][0-9]+)?$"},
                        "raw": {"type": "string", "minLength": 1},
                        "span_id": {"type": "string", "pattern": "^doc-[1-3]:p[0-9]{4}$"},
                    },
                },
            ],
        },
        "evidence": {
            "type": "object", "additionalProperties": False,
            "required": list(_EVIDENCE_FIELDS),
            "properties": {
                field: {
                    "type": "array", "uniqueItems": True,
                    "items": {"type": "string", "pattern": "^doc-[1-3]:p[0-9]{4}$"},
                }
                for field in _EVIDENCE_FIELDS
            },
        },
    },
}
