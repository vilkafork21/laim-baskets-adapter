"""Контракт monitoring_metric: версии, валидация и формула КМ.

Версии контракта:
  v1 — без assessment_mode (только qa), поднимается автоматически;
  v2 — готовый метод score (identity, accuracy, mean_criteria, all_criteria,
       majority, all_assessors); формула синтезируется из метода;
  v3 — формула записана явно (`formula`), method может быть "formula".
Нода принимает все три версии и работает с v3; отдаёт только v3.
"""

from __future__ import annotations

from copy import deepcopy

from . import formula
from .errors import MonitoringContractError
from .values import to_decimal

VERSION = "laim-monitoring-metric.v3"
ACCEPTED_VERSIONS = ("laim-monitoring-metric.v1", "laim-monitoring-metric.v2", VERSION)
UMR_VERSION = "laim-umr.v2"
ASSESSMENT_MODES = {"qa", "turn_with_history", "dialogue"}
ROLES = {"final_score", "criterion", "assessor_vote", "prediction", "target"}
MISSING = {"fail", "exclude_unit", "exclude_value", "zero"}
WEIGHT = "weight"
# Готовые методы v2 и допустимое число источников каждой роли.
METHOD_ROLES = {
    "identity": {"final_score": (1, 1)},
    "accuracy": {"prediction": (1, 1), "target": (1, 1)},
    "mean_criteria": {"criterion": (1, None)},
    "all_criteria": {"criterion": (1, None)},
    "majority": {"assessor_vote": (1, None)},
    "all_assessors": {"assessor_vote": (2, None)},
}
JUDGE_SCORE_SOURCE_ID = "assessment_score"


def require(mapping: dict, name: str, expected=None):
    if name not in mapping:
        raise MonitoringContractError(f"monitoring_metric не содержит {name}")
    value = mapping[name]
    if expected is not None and value not in expected:
        raise MonitoringContractError(f"Недопустимое {name}: {value!r}")
    return value


def source_name(source: dict) -> str:
    """Имя входа в формуле: явный name либо source_id."""
    return str(source.get("name") or source["source_id"])


def contract_formula(contract: dict) -> str:
    """Текст формулы КМ: явный `formula`, иначе синтез из готового метода v2."""
    explicit = contract.get("formula")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    scoring = contract["scoring"]
    method = scoring["method"]
    if method == "formula":
        raise MonitoringContractError("method=formula требует текст formula")
    by_role: dict[str, list[str]] = {}
    for source in scoring["sources"]:
        by_role.setdefault(source["role"], []).append(source_name(source))
    policy = scoring.get("missing_policy", "fail")

    def rows(names: list[str]) -> list[str]:
        return [f"fillna({name}, 0)" if policy == "zero" else name for name in names]

    if method == "identity":
        unit = rows(by_role["final_score"])[0]
    elif method == "accuracy":
        unit = f"{by_role['prediction'][0]} == {by_role['target'][0]}"
    elif method == "mean_criteria":
        names = rows(by_role["criterion"])
        unit = (
            f"avg({', '.join(names)})" if policy == "exclude_value" and len(names) > 1
            else f"({' + '.join(names)}) / {len(names)}" if len(names) > 1
            else names[0]
        )
    elif method in ("all_criteria", "all_assessors"):
        names = rows(by_role["criterion" if method == "all_criteria" else "assessor_vote"])
        unit = f"min({', '.join(names)})" if len(names) > 1 else names[0]
    else:  # majority
        declared = scoring.get("majority_denominator") == "declared"
        unit = f"majority({', '.join(by_role['assessor_vote'])}{', declared=True' if declared else ''})"
    weighted = contract.get("aggregation", {}).get("method") == "frequency_weighted_mean"
    return f"wmean({unit}, {WEIGHT})" if weighted else f"mean({unit})"


def _upgrade_contract(payload: dict) -> dict:
    """Любая принятая версия → v3 с явной формулой."""
    version = payload.get("contract_version")
    if version not in ACCEPTED_VERSIONS:
        raise MonitoringContractError(f"Неизвестная версия monitoring_metric: {version!r}")
    contract = deepcopy(payload)
    if version == ACCEPTED_VERSIONS[0]:
        if contract.get("status") == "computed" and contract.get("evaluation_unit") != "turn":
            raise MonitoringContractError("monitoring_metric.v1 dialogue нельзя восстановить без turn_index")
        contract["umr_version"] = UMR_VERSION
        contract["assessment_mode"] = "qa"
        if contract.get("status") == "not_evaluable":
            contract["status"] = "not_computable"
        contract.pop("evaluation_unit", None)
        contract.pop("group_column", None)
    if contract.get("status") == "computed" and not contract.get("formula"):
        try:
            contract["formula"] = contract_formula(contract)
        except (KeyError, TypeError):
            pass  # структура неполная — validate скажет, чего не хватает
    contract["contract_version"] = VERSION
    return contract


def validate_monitoring_metric(payload: object, *, require_computed: bool = True) -> dict:
    if not isinstance(payload, dict):
        raise MonitoringContractError("monitoring_metric должен быть JSON object")
    contract = _upgrade_contract(payload)
    if require(contract, "umr_version") != UMR_VERSION:
        raise MonitoringContractError(f"Неизвестная версия UMR: {contract.get('umr_version')!r}")
    status = require(contract, "status", {"computed", "not_computable"})
    if status != "computed":
        if require_computed:
            raise MonitoringContractError(
                f"monitoring_metric невычислим: {contract.get('reason', 'причина не указана')}"
            )
        return contract

    require(contract, "basket_id")
    require(contract, "name")
    if require(contract, "score_column") != "main_metric":
        raise MonitoringContractError("Единственная каноническая score-колонка: main_metric")
    require(contract, "assessment_mode", ASSESSMENT_MODES)

    scoring = require(contract, "scoring")
    if not isinstance(scoring, dict):
        raise MonitoringContractError("scoring должен быть object")
    method = require(scoring, "method", set(METHOD_ROLES) | {"formula"})
    sources = require(scoring, "sources")
    if not isinstance(sources, list) or not sources:
        raise MonitoringContractError("scoring.sources должен быть непустым списком")
    source_ids = set()
    names = set()
    role_counts: dict[str, int] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise MonitoringContractError("Каждый scoring source должен быть object")
        source_id = require(source, "source_id")
        if source_id in source_ids:
            raise MonitoringContractError(f"Повторяется source_id: {source_id}")
        source_ids.add(source_id)
        require(source, "column_name")
        role = require(source, "role", ROLES)
        role_counts[role] = role_counts.get(role, 0) + 1
        # Адаптер публикует значения уже нормализованными: числа и метки.
        require(source, "normalization", {"numeric", "label"})
        require(source, "polarity", {"direct"})
        name = source_name(source)
        if not name.isidentifier() or name in formula.HELPERS or name == WEIGHT:
            raise MonitoringContractError(f"Недопустимое имя входа формулы: {name!r}")
        if name in names:
            raise MonitoringContractError(f"Повторяется имя входа формулы: {name!r}")
        names.add(name)
    if method == "formula":
        if not isinstance(contract.get("formula"), str):
            raise MonitoringContractError("method=formula требует текст formula")
    else:
        expected_roles = METHOD_ROLES[method]
        if set(role_counts) != set(expected_roles):
            raise MonitoringContractError(
                f"Метод {method} требует роли {sorted(expected_roles)}, получено {sorted(role_counts)}"
            )
        for role, (minimum, maximum) in expected_roles.items():
            count = role_counts[role]
            if count < minimum or (maximum is not None and count > maximum):
                raise MonitoringContractError(f"Недопустимое число источников роли {role}: {count}")
    missing_policy = require(scoring, "missing_policy", MISSING)
    denominator = scoring.get("majority_denominator")
    if method == "majority" and denominator not in {"declared", "present"}:
        raise MonitoringContractError("majority требует denominator declared или present")
    if method != "majority" and denominator is not None:
        raise MonitoringContractError("majority_denominator допустим только для majority")
    aggregation = require(contract, "aggregation")
    reducer = require(aggregation, "method", {"mean", "frequency_weighted_mean"})
    weight_column = aggregation.get("weight_column")
    if reducer == "frequency_weighted_mean" and weight_column != "input_query_count":
        raise MonitoringContractError("Weighted mean требует input_query_count")
    if reducer == "mean" and weight_column is not None:
        raise MonitoringContractError("mean не должен объявлять weight_column")

    try:
        parsed = formula.parse(require(contract, "formula"))
    except formula.FormulaError as exc:
        raise MonitoringContractError(f"Формула КМ: {exc}") from exc
    unknown = [name for name in parsed.inputs if name not in names and name != WEIGHT]
    if unknown:
        raise MonitoringContractError(
            f"Формула ссылается на входы {unknown}, объявлены {sorted(names)} и {_WEIGHT}"
        )
    baseline = require(contract, "baseline")
    to_decimal(require(baseline, "value"), "baseline.value")
    to_decimal(require(baseline, "recomputed_value"), "baseline.recomputed_value")
    require(baseline, "scale", {"ratio", "raw"})
    validation = require(contract, "primary_validation")
    if validation.get("affects_monitoring") is not False:
        raise MonitoringContractError("Primary validation threshold не должен влиять на monitoring")
    return contract


def judge_score_contract(contract: dict) -> dict:
    """Контракт, в котором судья ставит готовый score единицы (assessment_score).

    Используется там, где формула отчёта неприменима: prediction не наблюдается
    в трейсах либо единица оценки — целый диалог. Ассесор и km-dynamic берут
    его из одного места, чтобы считать одно и то же.
    """
    result = deepcopy(contract)
    weighted = contract.get("aggregation", {}).get("method") == "frequency_weighted_mean"
    result["scoring"] = {
        "method": "identity",
        "sources": [{
            "source_id": JUDGE_SCORE_SOURCE_ID,
            "name": JUDGE_SCORE_SOURCE_ID,
            "column_name": "main_metric",
            "role": "final_score",
            "normalization": "numeric",
            "polarity": "direct",
        }],
        "missing_policy": contract["scoring"]["missing_policy"],
        "majority_denominator": None,
    }
    result["formula"] = (
        f"wmean({JUDGE_SCORE_SOURCE_ID}, {WEIGHT})" if weighted else f"mean({JUDGE_SCORE_SOURCE_ID})"
    )
    return result
