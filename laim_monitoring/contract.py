"""Контракт КМ — содержимое порта `monitoring_metric`.

Одна форма, одна версия. Контракт отвечает на четыре вопроса:

* **какая формула** — `formula`, как метрика определена в отчёте о валидации;
* **над чем** — `inputs`: имя в формуле, колонка UMR и размечает ли её судья
  (`judged`); вход с `judged=false` — наблюдаемый ответ агента из трейса;
* **что считается единицей** — `assessment_mode`: реплика, реплика с историей
  или диалог;
* **с чем сравнивать** — `baseline`: значение из отчёта, воспроизведённое
  пересчётом на эталонной корзине (`reconciliation` = match, иначе контракт
  не принимается).

Пример:

    {
      "contract_version": "laim-monitoring-metric.v3",
      "status": "computed",
      "basket_id": "CI09997438",
      "metric_name": "Accuracy",
      "assessment_mode": "qa",
      "formula": "mean(prediction == target)",
      "inputs": [
        {"name": "prediction", "column": "класс_output_answer", "judged": false},
        {"name": "target",     "column": "класс_metric",        "judged": true}
      ],
      "baseline": {"value": 0.9387, "recomputed_value": 0.93871, "reconciliation": "match"}
    }
"""

from __future__ import annotations

from copy import deepcopy

from . import formula
from .errors import MonitoringContractError
from .values import to_decimal

VERSION = "laim-monitoring-metric.v3"
ASSESSMENT_MODES = {"qa", "turn_with_history", "dialogue"}
WEIGHT = "weight"                  # зарезервированное имя: input_query_count единицы
JUDGE_SCORE_INPUT = "assessment_score"


def require(mapping: dict, name: str, expected=None):
    if name not in mapping:
        raise MonitoringContractError(f"monitoring_metric не содержит {name}")
    value = mapping[name]
    if expected is not None and value not in expected:
        raise MonitoringContractError(f"Недопустимое {name}: {value!r}")
    return value


def validate_monitoring_metric(payload: object, *, require_computed: bool = True) -> dict:
    if not isinstance(payload, dict):
        raise MonitoringContractError("monitoring_metric должен быть JSON object")
    version = payload.get("contract_version")
    if version != VERSION:
        raise MonitoringContractError(
            f"Версия monitoring_metric {version!r} не поддерживается, ожидается {VERSION}: "
            "обновите адаптер и ноды до одной версии пакета laim_monitoring"
        )
    contract = deepcopy(payload)
    status = require(contract, "status", {"computed", "not_computable"})
    if status != "computed":
        if require_computed:
            raise MonitoringContractError(
                f"monitoring_metric невычислим: {contract.get('reason', 'причина не указана')}"
            )
        return contract

    require(contract, "basket_id")
    if not str(contract.get("metric_name") or "").strip():
        raise MonitoringContractError("metric_name пуст")
    require(contract, "assessment_mode", ASSESSMENT_MODES)

    inputs = require(contract, "inputs")
    if not isinstance(inputs, list) or not inputs:
        raise MonitoringContractError("inputs должен быть непустым списком")
    names: set[str] = set()
    for item in inputs:
        if not isinstance(item, dict):
            raise MonitoringContractError("Каждый вход должен быть object {name, column, judged}")
        name = str(require(item, "name"))
        if not name.isidentifier() or name in formula.HELPERS or name == WEIGHT:
            raise MonitoringContractError(f"Недопустимое имя входа: {name!r}")
        if name in names:
            raise MonitoringContractError(f"Повторяется имя входа: {name!r}")
        names.add(name)
        if not str(require(item, "column") or "").strip():
            raise MonitoringContractError(f"Вход {name!r}: пустое имя колонки")
        if not isinstance(require(item, "judged"), bool):
            raise MonitoringContractError(f"Вход {name!r}: judged должен быть true/false")

    try:
        parsed = formula.parse(require(contract, "formula"))
    except formula.FormulaError as exc:
        raise MonitoringContractError(f"Формула КМ: {exc}") from exc
    unknown = [name for name in parsed.inputs if name not in names and name != WEIGHT]
    if unknown:
        raise MonitoringContractError(
            f"Формула ссылается на входы {unknown}, объявлены {sorted(names)} и {WEIGHT}"
        )

    baseline = require(contract, "baseline")
    to_decimal(require(baseline, "value"), "baseline.value")
    if baseline.get("reconciliation") != "match":
        raise MonitoringContractError(
            "baseline не воспроизведён пересчётом на эталонной корзине "
            f"(reconciliation={baseline.get('reconciliation')!r}): сравнивать с ним нельзя"
        )
    return contract


def uses_weight(contract: dict) -> bool:
    return WEIGHT in formula.parse(contract["formula"]).inputs


def judged_inputs(contract: dict) -> list[dict]:
    return [item for item in contract["inputs"] if item["judged"]]


def agent_inputs(contract: dict) -> list[dict]:
    return [item for item in contract["inputs"] if not item["judged"]]


def judge_score_contract(contract: dict) -> dict:
    """Контракт, где судья ставит готовый score единицы в main_metric.

    Применяется, когда формулу отчёта на мониторинге посчитать нельзя: ответ
    агента не наблюдается в трейсах либо единица — целый диалог. Ассесор и
    km-dynamic берут его отсюда, чтобы считать одно и то же.
    """
    result = deepcopy(contract)
    result["inputs"] = [{"name": JUDGE_SCORE_INPUT, "column": "main_metric", "judged": True}]
    result["formula"] = (
        f"wmean({JUDGE_SCORE_INPUT}, {WEIGHT})" if uses_weight(contract) else f"mean({JUDGE_SCORE_INPUT})"
    )
    return result
