"""Этап K: проверка цитат и детерминированный выбор КМ из отчёта валидации."""
from __future__ import annotations

import re
import math
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from itertools import combinations

import jsonschema

from ..errors import MeasurementPlanError
from ..llm.schemas import BASELINE_CANDIDATE_SCHEMA
from ..models import MeasurementPlan

_NUMBER = re.compile(
    r'[+-]?(?:[0-9]{1,3}(?:[ \xa0\u202f][0-9]{3})+|[0-9]+|(?=[.,][0-9]))'
    r'(?:[.,][0-9]+)?(?:[eE][+-]?[0-9]+)?\s*%?'
)
_PARAGRAPH = re.compile(r'p([0-9]{3,})')
_CANDIDATE_VALIDATOR = jsonschema.Draft202012Validator(BASELINE_CANDIDATE_SCHEMA)


def _name(text: str | None) -> str:
    return ' '.join((text or '').casefold().split())


def _matches(label: str | None, sheet: str) -> bool:
    left, right = _name(label), _name(sheet)
    return bool(left and right and (left in right or right in left))


def _parse(raw: str, default_scale: str) -> tuple[Decimal, str]:
    token = raw.strip()
    if not _NUMBER.fullmatch(token):
        raise ValueError('raw не является одним конечным числом с необязательным %')
    scale = 'percent' if token.endswith('%') else default_scale
    try:
        value = Decimal(re.sub(r'\s+', '', token.rstrip('%')).replace(',', '.'))
    except InvalidOperation as exc:
        raise ValueError('raw не разбирается как число') from exc
    if not value.is_finite():
        raise ValueError('число должно быть конечным')
    if not math.isfinite(float(value)) or (value and float(value) == 0):
        raise ValueError('число не представимо в числовом поле JSON без переполнения или потери')
    return value, scale


def _offset(raw: str, paragraph: str) -> int:
    """Дословная цитата без вырезания цифр, знака, экспоненты или процента."""
    for match in re.finditer(re.escape(raw), paragraph):
        before, after = paragraph[:match.start()], paragraph[match.end():]
        if re.search(r'[0-9]$|[0-9][.,]$|[+-]$', before):
            continue
        if re.match(r'[0-9]|[.,][0-9]|[eE][+-]?[0-9]', after):
            continue
        if not raw.rstrip().endswith('%') and re.match(r'\s*%', after):
            continue
        return match.start()
    raise ValueError('raw не найден дословно в указанном абзаце с границами числа')


def _convert(value: Decimal, source: str, target: str) -> Decimal:
    if source == target:
        return value
    if source == 'percent' and target == 'ratio':
        return value.scaleb(-2)
    if source == 'ratio' and target == 'percent':
        return value.scaleb(2)
    raise MeasurementPlanError(
        'Шкала объявленной КМ несовместима со шкалой построчной оценки',
        reported_scale=source, score_scale=target,
    )


def _comparable(candidate: dict) -> tuple[Decimal, Decimal, str]:
    value = Decimal(candidate['parsed_value'])
    quantum = Decimal(1).scaleb(value.as_tuple().exponent)
    scale = candidate['scale']
    if scale == 'percent':
        return value / 100, quantum / 100, 'ratio'
    return value, quantum, scale


def _equivalent(left: dict, right: dict) -> bool:
    a, qa, sa = _comparable(left)
    b, qb, sb = _comparable(right)
    # Проверяются все пары: грубое округление не склеивает разные точные числа.
    return sa == sb and abs(a - b) <= max(qa, qb) / 2


@dataclass(frozen=True)
class BaselineResult:
    state: str
    value: Decimal | None = None
    raw: str | None = None
    paragraph: str | None = None
    metric_name: str = ''
    scale: str = 'raw'
    candidates: tuple[dict, ...] = ()
    rejected: tuple[dict, ...] = ()
    threshold: Decimal | None = None
    comparator: str | None = None
    threshold_raw: str | None = None
    threshold_paragraph: str | None = None
    reason_code: str | None = None
    reason: str | None = None
    warnings: tuple[dict, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            'state': self.state,
            'value': float(self.value) if self.value is not None else None,
            'raw': self.raw, 'paragraph': self.paragraph,
            'metric_name': self.metric_name or None, 'scale': self.scale,
            'candidates': list(self.candidates), 'rejected': list(self.rejected),
            'threshold': None if self.threshold is None else {
                'value': float(self.threshold), 'scale': self.scale,
                'raw': self.threshold_raw, 'paragraph': self.threshold_paragraph,
                'comparator': self.comparator,
            },
            'reason_code': self.reason_code, 'reason': self.reason,
            'warnings': list(self.warnings),
        }


def failed_baseline(observation: str, *, ambiguous: bool = False,
                    candidates=(), rejected=()) -> BaselineResult:
    code = 'ambiguous_baseline' if ambiguous else 'official_baseline_missing'
    return BaselineResult(
        state='ambiguous' if ambiguous else 'not_declared',
        candidates=tuple(candidates), rejected=tuple(rejected), reason_code=code,
        reason=(
            'Этап K. Ожидалось: одно подтверждённое значение ключевой метрики '
            'для выбранного листа в отчёте о валидации. '
            f'Получено: {observation}. Действие: проверьте baseline.candidates '
            'и baseline.rejected, уточните отчёт о валидации или настройку sheet_name '
            'и повторите запуск.'
        ),
    )


def select_baseline(candidates: list[dict], paragraphs: tuple[str, ...], *,
                    selected_sheet: str, metric_name: str = '',
                    scale: str | None = None) -> BaselineResult:
    """Возвращает результат K независимо от успеха S; не доверяет state модели."""
    verified, rejected = [], []
    default_scale = scale or 'raw'
    for candidate in candidates:
        try:
            _CANDIDATE_VALIDATOR.validate(candidate)
            raw = candidate['raw']
            ref = _PARAGRAPH.fullmatch(candidate['paragraph'])
            index = int(ref.group(1)) - 1 if ref else -1
            if not 0 <= index < len(paragraphs):
                raise ValueError('paragraph не указывает на существующий абзац отчёта')
            value, value_scale = _parse(raw, default_scale)
            offset = _offset(raw, paragraphs[index])
        except (ValueError, jsonschema.ValidationError) as exc:
            rejected.append({'candidate': candidate, 'reason': str(exc)})
            continue
        verified.append({**candidate, 'parsed_value': str(value), 'scale': value_scale,
                         'position': offset, 'selected': False})

    slices = [item for item in verified if item['kind'] == 'slice_value']
    matched = [item for item in slices if _matches(item['slice_label'], selected_sheet)]
    values = matched or [item for item in verified if item['kind'] == 'value']
    if not values:
        return failed_baseline(
            f'проверенных упоминаний {len(verified)}, отклонённых {len(rejected)}; '
            + ('найдены только значения других срезов' if slices else 'итоговое значение не найдено'),
            ambiguous=bool(slices), candidates=verified, rejected=rejected,
        )
    key = [item for item in values if item['is_key_metric']]
    values = key or values
    named = [item for item in values if _name(item['metric_name']) == _name(metric_name)]
    values = named or values

    # В строке таблицы сначала значение, затем ДИ. Порядок ответа LLM не влияет.
    first = {}
    for item in sorted(values, key=lambda entry: (int(entry['paragraph'][1:]), entry['position'])):
        first.setdefault((item['paragraph'], _name(item['metric_name'])), item)
    values = list(first.values())
    if any(not _equivalent(left, right) for left, right in combinations(values, 2)):
        return failed_baseline(
            f'{len(values)} разных кандидатов на итоговую КМ', ambiguous=True,
            candidates=verified, rejected=rejected,
        )
    chosen = min(values, key=lambda entry: (_comparable(entry)[1],
                                          int(entry['paragraph'][1:]), entry['position']))
    chosen['selected'] = True
    value = Decimal(chosen['parsed_value'])
    value_scale = chosen['scale']

    thresholds = [item for item in verified if item['kind'] == 'threshold'
                  and _name(item['metric_name']) == _name(chosen['metric_name'])
                  and (not item['slice_label'] or _matches(item['slice_label'], selected_sheet))]
    sliced_thresholds = [item for item in thresholds if item['slice_label']]
    thresholds = sliced_thresholds or thresholds
    threshold = comparator = threshold_raw = threshold_paragraph = None
    warnings = []
    if thresholds:
        same = all(_equivalent(left, right)
                   and (left.get('comparator') or '>=') == (right.get('comparator') or '>=')
                   for left, right in combinations(thresholds, 2))
        if same:
            selected = min(thresholds, key=lambda entry: _comparable(entry)[1])
            try:
                threshold = _convert(Decimal(selected['parsed_value']), selected['scale'], value_scale)
                comparator = selected.get('comparator') or '>='
                threshold_raw, threshold_paragraph = selected['raw'], selected['paragraph']
            except MeasurementPlanError:
                same = False
        if not same:
            warnings.append({'code': 'threshold_ambiguous', 'message':
                'Этап K. Ожидалось: один порог в согласованной шкале. Получено: '
                'несогласованные пороги. Действие: уточните порог в отчёте; '
                'вердикт по порогу не опубликован.'})
    return BaselineResult(
        state='declared', value=value, raw=chosen['raw'], paragraph=chosen['paragraph'],
        metric_name=chosen['metric_name'], scale=value_scale,
        candidates=tuple(verified), rejected=tuple(rejected), threshold=threshold,
        comparator=comparator, threshold_raw=threshold_raw,
        threshold_paragraph=threshold_paragraph, warnings=tuple(warnings),
    )


def attach_baseline(plan: MeasurementPlan, baseline: BaselineResult) -> MeasurementPlan:
    """Сборка K + S: не меняет метод оценки, источники или домен плана S."""
    if baseline.state != 'declared':
        return plan
    value = _convert(baseline.value, baseline.scale, plan.scale)
    threshold = None if baseline.threshold is None else _convert(
        baseline.threshold, baseline.scale, plan.scale)
    return replace(
        plan, reported_value=value, reported_raw=baseline.raw,
        precision=max(0, -value.as_tuple().exponent), threshold=threshold,
        comparator=baseline.comparator,
        evidence={**plan.evidence, 'baseline': (f'{baseline.paragraph}: {baseline.raw}',)},
    )
