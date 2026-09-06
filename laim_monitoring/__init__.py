"""Общий пакет нод мониторинга LAIM: контракт КМ, единицы оценки, формула.

Вендорится в каждую ноду целиком (Sber DS деплоит zip) из одного источника —
`laim/monitoring/shared/laim_monitoring`. Версия печатается нодами в выходах.
"""

__version__ = "3.0.0"

from .contract import (
    JUDGE_SCORE_INPUT,
    VERSION,
    agent_inputs,
    judge_score_contract,
    judged_inputs,
    validate_monitoring_metric,
)
from .drift import prepare_drift_frames
from .errors import MonitoringContractError
from .formula import FormulaError, parse as parse_formula
from .scoring import aggregate_main_metric, broadcast_scores, evaluate_formula, unit_scores
from .units import normalize_umr, unitize

__all__ = [
    "FormulaError",
    "JUDGE_SCORE_INPUT",
    "MonitoringContractError",
    "VERSION",
    "__version__",
    "agent_inputs",
    "aggregate_main_metric",
    "broadcast_scores",
    "evaluate_formula",
    "judge_score_contract",
    "judged_inputs",
    "normalize_umr",
    "parse_formula",
    "prepare_drift_frames",
    "unit_scores",
    "unitize",
    "validate_monitoring_metric",
]
