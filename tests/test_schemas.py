"""Схемы ответов LLM: валидные примеры проходят, цитаты стоят первыми."""
from __future__ import annotations


import jsonschema
import pytest


from laim_basket.llm.schemas import LAYOUT_SCHEMA, METRIC_SCHEMA

LAYOUT_EXAMPLE = {
    "quotes": {"columns": ["q - запрос клиента"]},
    "sheet_name": "Лист1",
    "header_rows": [1],
    "roles": {
        "query_id": None, "session_id": None,
        "input_query": "A", "output_answer": "B",
        "scenario": None, "assessor_id": None,
        "reference_answers": [],
    },
    "grouping": {"kind": "none", "column": None},
    "dialogue_blob": None,
    "weight_column": None,
}

METRIC_EXAMPLE = {
    "quotes": {"metric": ["Ключевая метрика Accuracy равна 0.93"]},
    "metric_name": "Accuracy",
    "method": "identity",
    "sources": [{"column_id": "C", "role": "final_score",
                  "normalization": "numeric", "polarity": "direct"}],
    "reducer": "mean",
    "missing_policy": "exclude_unit",
    "majority_denominator": None,
    "scale": "ratio",
    "reported_value": {"state": "declared", "value": 0.93, "raw": "0.93"},
    "threshold": 0.9,
    "comparator": ">=",
}


def test_examples_validate():
    jsonschema.validate(LAYOUT_EXAMPLE, LAYOUT_SCHEMA)
    jsonschema.validate(METRIC_EXAMPLE, METRIC_SCHEMA)


def test_quotes_come_first_for_grounding():
    assert next(iter(LAYOUT_SCHEMA["properties"])) == "quotes"
    assert next(iter(METRIC_SCHEMA["properties"])) == "quotes"


def test_blob_dialogue_variant_validates():
    example = dict(
        LAYOUT_EXAMPLE,
        roles=dict(LAYOUT_EXAMPLE["roles"], output_answer=None),
        grouping={"kind": "blob_row", "column": None},
        dialogue_blob={"column": "B", "container": "python_list",
                        "question_marker": "КЛИЕНТ:", "answer_marker": "АГЕНТ:"},
    )
    jsonschema.validate(example, LAYOUT_SCHEMA)


def test_value_map_normalization_validates():
    example = dict(
        METRIC_EXAMPLE,
        sources=[{"column_id": "C", "role": "criterion",
                   "normalization": {"да": 1, "нет": 0}, "polarity": "direct"},
                  {"column_id": "D", "role": "criterion",
                   "normalization": "numeric", "polarity": "direct"}],
        method="mean_criteria",
    )
    jsonschema.validate(example, METRIC_SCHEMA)


def test_wrong_method_enum_rejected():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(METRIC_EXAMPLE, method="median"), METRIC_SCHEMA)
