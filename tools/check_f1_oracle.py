"""Дифференциальная проверка F1 через sklearn; sklearn — только инструмент QA.

Не входит в runtime ZIP. Детерминированные случайные выборки, веса, список
классов, отсутствующие классы и варианты zero_division. Никаких пропусков
тестов при отсутствии sklearn: установка зависимости — обязанность стенда.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from decimal import Decimal
from pathlib import Path

import sklearn
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laim_basket.metric.nonadditive import aggregate  # noqa: E402
from laim_basket.models import MeasurementPlan  # noqa: E402


def check() -> dict:
    rng = random.Random(20260916)
    count, worst = 0, 0.0
    for average in ("binary", "micro", "macro", "weighted"):
        for repetition in range(100):
            domain = ["0", "1"] if average == "binary" else ["a", "b", "c", "d"]
            n = rng.randint(1, 90)
            actual = [rng.choice(domain) for _ in range(n)]
            predicted = [rng.choice(domain) for _ in range(n)]
            weights = [rng.randint(1, 15) for _ in range(n)]
            for weighted in (False, True):
                for zero in (0, 1):
                    selected = (
                        None
                        if average == "binary" or repetition % 3 == 0
                        else [*domain[: (repetition % len(domain)) + 1], "absent"]
                    )
                    options = {
                        "average": average,
                        "labels": selected,
                        "positive_label": "1" if average == "binary" else None,
                        "zero_division": zero,
                    }
                    sources = tuple(
                        {
                            "column_id": column,
                            "role": role,
                            "normalization": "label",
                            "polarity": "direct",
                        }
                        for column, role in [("P", "prediction"), ("T", "target")]
                    )
                    plan = MeasurementPlan(
                        basket_id="synthetic",
                        metric_name="F1",
                        assessment_mode="qa",
                        method="classification_f1",
                        sources=sources,
                        missing_policy="exclude_unit",
                        majority_denominator=None,
                        reducer="frequency_weighted_mean" if weighted else "mean",
                        threshold=None,
                        comparator=None,
                        scale="ratio",
                        precision=0,
                        reported_value=None,
                        reported_raw=None,
                        evidence={},
                        metric_options=options,
                    )
                    records = [
                        {"values": {"P": p, "T": t}, "weight": Decimal(w), "rows": [i]}
                        for i, (p, t, w) in enumerate(zip(predicted, actual, weights))
                    ]
                    observed = float(aggregate(records, plan)[0])
                    expected = f1_score(
                        actual,
                        predicted,
                        average=average,
                        labels=selected,
                        pos_label="1" if average == "binary" else 1,
                        sample_weight=weights if weighted else None,
                        zero_division=zero,
                    )
                    error = abs(observed - expected)
                    assert math.isclose(observed, expected, abs_tol=1e-12), (
                        options,
                        actual,
                        predicted,
                        weights,
                        observed,
                        expected,
                    )
                    worst = max(worst, error)
                    count += 1
    # Поддержка выбранных классов равна нулю: weighted следует документированной
    # реализации sklearn, а не делит на ноль или произвольно отбрасывает классы.
    return {
        "cases": count,
        "failures": 0,
        "max_absolute_error": worst,
        "seed": 20260916,
        "sklearn_version": sklearn.__version__,
        "execution_mode": "synthetic-differential-NOT-LLM",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
