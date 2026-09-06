"""Формула КМ: одно выражение над колонками корзины.

Принцип: метрика не выбирается из реестра, а записывается так, как она
определена в отчёте о валидации, — выражением над построчной разметкой
корзины. Одна и та же строка формулы:

* пересчитывает baseline на эталонной корзине (разметка человека) и обязана
  дать число из отчёта — это доказательство, что формула понята верно;
* считает КМ на мониторинге по разметке судьи и наблюдаемому ответу агента.

Язык формулы намеренно мал и читается без документации:

    имена       — входы контракта (например target, prediction, полнота, weight)
    операции    — + - * /, сравнения == != < <= > >=, and / or / not, скобки
    агрегаты    — mean(x), wmean(x, w), sum(x), count(x)
    построчные  — avg(a, b, ...), min(a, b, ...), max(a, b, ...), abs(x),
                  fillna(x, значение), majority(a, b, ..., declared=False)
    классы      — precision(prediction, target, average), recall(...), f1(...)
                  average: "macro" | "micro" | "weighted" | метка класса

Примеры:
    mean(prediction == target)                 доля верных ответов (accuracy)
    wmean(prediction == target, weight)        то же, взвешенно по частоте
    mean((полнота + точность + ясность) / 3)   среднее критериев
    mean(min(полнота, точность))               «все критерии выполнены»
    mean(majority(а1, а2, а3))                 решение большинства асессоров
    f1(prediction, target, "macro")            macro-F1 по классам
    mean(оценка >= 4)                          доля диалогов с оценкой не ниже 4

Пустые значения (NaN) пропускаются агрегатами и распространяются через
построчные операции: строка без нужной разметки не участвует в среднем. Если
отчёт считает пропуск нулём, формула говорит это явно через fillna.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


class FormulaError(ValueError):
    """Формула не разбирается, ссылается на неизвестный вход или не даёт число."""


_BINARY = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_COMPARE = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
_AGGREGATES = ("mean", "wmean", "sum", "count", "precision", "recall", "f1")
_ROWWISE = ("avg", "min", "max", "abs", "fillna", "majority")
HELPERS = _AGGREGATES + _ROWWISE
RESERVED = ("weight",)


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _label(value: object) -> object:
    """Метки сравниваются как нормализованный текст, числа — как числа."""
    if _blank(value):
        return np.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return " ".join(text.casefold().replace("ё", "е").split())


def _numeric(series: pd.Series, name: str) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    if series.notna().sum() != result.notna().sum():
        raise FormulaError(f"{name}: ожидались числа, найдены текстовые значения")
    return result.astype("float64")


def _series(value: object, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    return pd.Series(value, index=index, dtype="float64")


def _compare(op: str, left: object, right: object, index: pd.Index) -> object:
    scalar = not isinstance(left, pd.Series) and not isinstance(right, pd.Series)
    a = _series(left, index).map(_label) if not scalar else _label(left)
    b = _series(right, index).map(_label) if not scalar else _label(right)
    if scalar:
        if _blank(a) or _blank(b):
            return np.nan
        return float(_apply_compare(op, a, b))
    a, b = _series(a, index), _series(b, index)
    mask = a.isna() | b.isna()
    result = pd.Series(
        [np.nan if m else float(_apply_compare(op, x, y)) for x, y, m in zip(a, b, mask)],
        index=index, dtype="float64",
    )
    return result


def _apply_compare(op: str, a: object, b: object) -> bool:
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if isinstance(a, str) or isinstance(b, str):
        raise FormulaError(f"сравнение {op} применимо только к числам")
    return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]


def _bool(value: object, index: pd.Index) -> pd.Series:
    series = _series(value, index)
    if series.dropna().isin([0.0, 1.0]).all():
        return series.astype("float64")
    raise FormulaError("and / or / not применимы только к результатам сравнений (0/1)")


# --------------------------------------------------------------------------- helpers

def _mean(x: object) -> float:
    if isinstance(x, pd.Series):
        return float(x.mean()) if x.notna().any() else np.nan
    return float(x)


def _wmean(x: object, w: object) -> float:
    if not isinstance(x, pd.Series) or not isinstance(w, pd.Series):
        raise FormulaError("wmean ожидает две колонки: значение и вес")
    w = _numeric(w, "weight")
    if (w.dropna() <= 0).any():
        raise FormulaError("wmean: вес должен быть положительным")
    mask = x.notna() & w.notna()
    if not mask.any():
        return np.nan
    return float((x[mask] * w[mask]).sum() / w[mask].sum())


def _sum(x: object) -> float:
    return float(x.sum()) if isinstance(x, pd.Series) else float(x)


def _count(x: object) -> float:
    return float(x.notna().sum()) if isinstance(x, pd.Series) else 1.0


def _rowwise(name: str, values: list[object], index: pd.Index) -> pd.Series:
    frame = pd.concat([_series(v, index) for v in values], axis=1)
    if name == "avg":
        return frame.mean(axis=1, skipna=True)
    if name == "min":
        return frame.min(axis=1, skipna=False)
    return frame.max(axis=1, skipna=False)


def _majority(votes: list[object], index: pd.Index, declared: bool = False) -> pd.Series:
    frame = pd.concat([_series(v, index) for v in votes], axis=1)
    if not frame.stack().dropna().isin([0.0, 1.0]).all():
        raise FormulaError("majority ожидает голоса 0/1")
    present = frame.notna().sum(axis=1)
    positives = frame.sum(axis=1, skipna=True)
    denominator = pd.Series(float(len(votes)), index=index) if declared else present.astype(float)
    result = pd.Series(np.nan, index=index, dtype="float64")
    result[positives * 2 > denominator] = 1.0
    result[(positives * 2 < denominator) & (present > 0)] = 0.0
    return result


def _class_metric(kind: str, prediction: object, target: object, average: object, index: pd.Index) -> float:
    pred = _series(prediction, index).map(_label)
    true = _series(target, index).map(_label)
    mask = pred.notna() & true.notna()
    pred, true = pred[mask], true[mask]
    if pred.empty:
        return np.nan
    classes = sorted(set(true.tolist()) | set(pred.tolist()), key=str)
    tp = {c: float(((pred == c) & (true == c)).sum()) for c in classes}
    fp = {c: float(((pred == c) & (true != c)).sum()) for c in classes}
    fn = {c: float(((pred != c) & (true == c)).sum()) for c in classes}

    def value(t: float, p: float, n: float) -> float:
        precision = t / (t + p) if t + p else 0.0
        recall = t / (t + n) if t + n else 0.0
        if kind == "precision":
            return precision
        if kind == "recall":
            return recall
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    if average == "micro":
        return value(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    support = {c: float((true == c).sum()) for c in classes}
    present = [c for c in classes if support[c] > 0]
    if average == "macro":
        return float(np.mean([value(tp[c], fp[c], fn[c]) for c in present]))
    if average == "weighted":
        total = sum(support[c] for c in present)
        return float(sum(value(tp[c], fp[c], fn[c]) * support[c] / total for c in present))
    label = _label(average)
    if label not in classes:
        raise FormulaError(f"{kind}: класс {average!r} не встречается в данных")
    return value(tp[label], fp[label], fn[label])


# --------------------------------------------------------------------------- evaluator

@dataclass(frozen=True)
class Formula:
    text: str
    tree: ast.Expression
    inputs: tuple[str, ...]

    def evaluate(self, columns: dict[str, pd.Series]) -> float:
        """Одно число по колонкам единиц оценки."""
        missing = [name for name in self.inputs if name not in columns]
        if missing:
            raise FormulaError(
                f"формула ссылается на входы {missing}, доступны {sorted(columns)}"
            )
        index = next(iter(columns.values())).index if columns else pd.Index([])
        result = _eval(self.tree.body, columns, index)
        if isinstance(result, pd.Series):
            raise FormulaError(
                "формула должна давать одно число: оберни построчное выражение в mean(...)"
            )
        return float(result)

    def unit_expression(self) -> "Formula | None":
        """Построчная часть формулы вида mean(E) / wmean(E, w); иначе None."""
        node = self.tree.body
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("mean", "wmean")
            and node.args
        ):
            inner = ast.Expression(body=node.args[0])
            ast.fix_missing_locations(inner)
            return Formula(ast.unparse(node.args[0]), inner, _names(inner))
        return None

    def evaluate_rows(self, columns: dict[str, pd.Series]) -> pd.Series:
        """Построчный результат (для unit_expression)."""
        index = next(iter(columns.values())).index
        return _series(_eval(self.tree.body, columns, index), index)


def parse(text: object) -> Formula:
    if not isinstance(text, str) or not text.strip():
        raise FormulaError("формула пуста")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"формула не разбирается: {exc.msg}") from exc
    for node in ast.walk(tree):
        _check_node(node)
    return Formula(text.strip(), tree, _names(tree))


def _names(tree: ast.AST) -> tuple[str, ...]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in HELPERS and node.id not in names:
            names.append(node.id)
    return tuple(names)


def _check_node(node: ast.AST) -> None:
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Call,
        ast.Name, ast.Constant, ast.Load, ast.keyword,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.Not, ast.And, ast.Or,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    )
    if not isinstance(node, allowed):
        raise FormulaError(f"недопустимая конструкция: {type(node).__name__}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in HELPERS:
            raise FormulaError("вызывать можно только функции формулы: " + ", ".join(HELPERS))
    if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, str, bool)):
        raise FormulaError(f"недопустимая константа: {node.value!r}")


def _eval(node: ast.AST, columns: dict[str, pd.Series], index: pd.Index) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return columns[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, columns, index)
        if isinstance(node.op, ast.USub):
            return -operand
        return 1.0 - _bool(operand, index)
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, columns, index)
        right = _eval(node.right, columns, index)
        op = _BINARY[type(node.op)]
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        return left / right
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise FormulaError("цепочки сравнений вида a < b < c не поддерживаются")
        left = _eval(node.left, columns, index)
        right = _eval(node.comparators[0], columns, index)
        return _compare(_COMPARE[type(node.ops[0])], left, right, index)
    if isinstance(node, ast.BoolOp):
        parts = [_bool(_eval(value, columns, index), index) for value in node.values]
        frame = pd.concat(parts, axis=1)
        if isinstance(node.op, ast.And):
            return frame.min(axis=1, skipna=False)
        return frame.max(axis=1, skipna=False)
    if isinstance(node, ast.Call):
        return _call(node, columns, index)
    raise FormulaError(f"недопустимая конструкция: {type(node).__name__}")


def _call(node: ast.Call, columns: dict[str, pd.Series], index: pd.Index) -> object:
    name = node.func.id
    args = [_eval(arg, columns, index) for arg in node.args]
    kwargs = {kw.arg: _eval(kw.value, columns, index) for kw in node.keywords}
    if name == "mean" and len(args) == 1 and not kwargs:
        return _mean(args[0])
    if name == "wmean" and len(args) == 2 and not kwargs:
        return _wmean(args[0], args[1])
    if name == "sum" and len(args) == 1 and not kwargs:
        return _sum(args[0])
    if name == "count" and len(args) == 1 and not kwargs:
        return _count(args[0])
    if name in ("avg", "min", "max") and len(args) >= 1 and not kwargs:
        return _rowwise(name, args, index)
    if name == "abs" and len(args) == 1 and not kwargs:
        return abs(args[0])
    if name == "fillna" and len(args) == 2 and not kwargs:
        return _series(args[0], index).fillna(float(args[1]))
    if name == "majority" and args and set(kwargs) <= {"declared"}:
        return _majority(args, index, declared=bool(kwargs.get("declared", False)))
    if name in ("precision", "recall", "f1") and len(args) in (2, 3) and set(kwargs) <= {"average"}:
        average = args[2] if len(args) == 3 else kwargs.get("average", "macro")
        return _class_metric(name, args[0], args[1], average, index)
    raise FormulaError(f"{name}: неверное число аргументов")
