"""Общий контракт мониторинга LAIM: формула КМ, единицы оценки, агрегация.

Пакет вендорится в каждую ноду (Sber DS деплоит zip) из одного источника —
`laim/monitoring/shared/laim_monitoring`. Версия печатается нодами в выходах,
чтобы рассинхрон копий был виден в отчёте.
"""

__version__ = "3.0.0"

from .contract import (
    JUDGE_SCORE_SOURCE_ID,
    VERSION,
    contract_formula,
    judge_score_contract,
    source_name,
    validate_monitoring_metric,
)
from .drift import prepare_drift_frames
from .errors import MonitoringContractError
from .formula import FormulaError, parse as parse_formula
from .scoring import aggregate_main_metric, broadcast_scores, formula_columns, score_units, unit_scores
from .units import normalize_umr, unitize

__all__ = [
    "FormulaError",
    "JUDGE_SCORE_SOURCE_ID",
    "MonitoringContractError",
    "VERSION",
    "__version__",
    "aggregate_main_metric",
    "broadcast_scores",
    "contract_formula",
    "formula_columns",
    "judge_score_contract",
    "normalize_umr",
    "parse_formula",
    "prepare_drift_frames",
    "score_units",
    "source_name",
    "unit_scores",
    "unitize",
    "validate_monitoring_metric",
]
