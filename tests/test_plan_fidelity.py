"""План подключения не должен переинтерпретироваться по оформлению книги."""
from __future__ import annotations

from copy import deepcopy

import pytest

from conftest import layout_answer, metric_answer, source
from helpers import make_workbook
from laim_basket.errors import LayoutError, NotEvaluableError
from laim_basket.metric.resolve import resolve_measurement_plan
from laim_basket.reading.xlsx_reader import read_workbook
from laim_basket.resolve import resolve_layout
from laim_basket.transform.canon import build_canon
from laim_basket.transform.grouping import apply_grouping


def materialize(tmp_path, proposal, merges=()):
    path = tmp_path / 'basket.xlsx'
    make_workbook(path, {'Лист1': {'rows': [
        ['q', 'a', 'score', 'category'],
        ['q1', 'a1', 1, 'общая'], ['q2', 'a2', 1, None], ['q3', 'a3', 0, 'другая'],
    ], 'merges': merges}})
    sheets = read_workbook(path)
    sheet = sheets['Лист1']
    layout = resolve_layout(proposal, sheets, 'CI09000001', '', frozenset())
    grouped = apply_grouping(sheet, layout.region, layout.transform_config())
    frame, _ = build_canon(grouped, layout.region, layout.transform_config())
    return layout, frame, sheet


def test_decorative_merge_does_not_change_explicit_grouping(tmp_path):
    proposal = layout_answer()
    original = deepcopy(proposal)
    layout, frame, _ = materialize(tmp_path, proposal, ['D2:D3'])
    assert layout.grouping['kind'] == 'none'
    assert 'reference_group_id' not in frame
    assert proposal == original


def test_merged_grouping_requires_explicit_anchor(tmp_path):
    with pytest.raises(LayoutError, match='column'):
        materialize(tmp_path, layout_answer(grouping={'kind': 'merged_rows', 'column': None}), ['D2:D3'])


def test_merged_grouping_uses_only_selected_anchor(tmp_path):
    proposal = layout_answer(grouping={'kind': 'merged_rows', 'column': 'A'})
    layout, frame, _ = materialize(tmp_path, proposal, ['A2:A3', 'D2:D4'])
    assert layout.grouping['column'] == 'q'
    assert frame.reference_group_id.tolist() == ['group-0', 'group-0', 'group-1']


@pytest.mark.parametrize('proposal', [
    metric_answer(method='mean_criteria', sources=[source('C', 'criterion')]),
    metric_answer(sources=[source('C', 'final_score'), source('D', 'criterion')]),
])
def test_metric_method_and_sources_are_not_silently_rewritten(tmp_path, proposal):
    layout, frame, sheet = materialize(tmp_path, layout_answer())
    with pytest.raises(NotEvaluableError):
        resolve_measurement_plan(proposal, layout, frame, sheet)


def test_decorative_merge_cannot_change_dialogue_metric(tmp_path):
    from laim_basket.metric.engine import evaluate

    results = []
    for decoration in (False, True):
        path = tmp_path / 'dialogues.xlsx'
        make_workbook(path, {'Лист1': {'rows': [
            ['s', 'q', 'a', 'score', 'category'],
            ['d1', 'q1', 'a1', 1, 'общая'], [None, 'q2', 'a2', None, None],
            ['d2', 'q3', 'a3', 1, None], [None, 'q4', 'a4', None, None],
            ['d3', 'q5', 'a5', 0, 'другая'],
        ], 'merges': ['A2:A3', 'A4:A5', 'D2:D3', 'D4:D5'] + (['E2:E5'] if decoration else [])}})
        sheets = read_workbook(path)
        proposal = layout_answer(grouping={'kind': 'merged_rows', 'column': 'A'})
        proposal['roles'].update(session_id='A', input_query='B', output_answer='C')
        layout = resolve_layout(proposal, sheets, 'CI09000001', '', frozenset())
        sheet = sheets['Лист1']
        frame, _ = build_canon(apply_grouping(sheet, layout.region, layout.transform_config()),
                               layout.region, layout.transform_config())
        plan = resolve_measurement_plan(metric_answer(assessment_mode='dialogue', sources=[source('D', 'final_score')]),
                                        layout, frame, sheet)
        _, km = evaluate(frame, layout, plan)
        results.append((km['coverage']['total_units'], km['recomputed_value']))
    assert results == [(3, 2 / 3), (3, 2 / 3)]


@pytest.mark.parametrize('route_source', [None, {'envelope': 'outgoing', 'field': 'receiver'},
                                       {'envelope': 'incoming', 'field': 'message', 'part_index': 2}])
def test_fipa_route_requires_and_preserves_explicit_source(tmp_path, route_source):
    from conftest import evaluation_answer

    layout, frame, sheet = materialize(tmp_path, layout_answer())
    evaluation = evaluation_answer(prediction_observable='route_label', observation_profile='fipa_external_reply_v1',
                                   external_party='external')
    if route_source is not None:
        evaluation['route_source'] = route_source
    proposal = metric_answer(evaluation=evaluation)
    if route_source is None:
        with pytest.raises(NotEvaluableError, match='route_source'):
            resolve_measurement_plan(proposal, layout, frame, sheet)
    else:
        plan = resolve_measurement_plan(proposal, layout, frame, sheet)
        assert plan.evaluation['route_source'] == route_source


def test_one_merged_score_cannot_label_two_distinct_dialogues(tmp_path):
    path = tmp_path / 'shared_score.xlsx'
    make_workbook(path, {'Лист1': {'rows': [
        ['s', 'q', 'a', 'score'], ['d1', 'q1', 'a1', 1], [None, 'q2', 'a2', None],
        ['d2', 'q3', 'a3', None], [None, 'q4', 'a4', None],
    ], 'merges': ['A2:A3', 'A4:A5', 'D2:D5']}})
    sheets = read_workbook(path)
    proposal = layout_answer(grouping={'kind': 'merged_rows', 'column': 'A'})
    proposal['roles'].update(session_id='A', input_query='B', output_answer='C')
    layout = resolve_layout(proposal, sheets, 'CI09000001', '', frozenset())
    sheet = sheets['Лист1']
    frame, _ = build_canon(apply_grouping(sheet, layout.region, layout.transform_config()),
                           layout.region, layout.transform_config())
    with pytest.raises(NotEvaluableError, match='несколько dialogue'):
        resolve_measurement_plan(metric_answer(assessment_mode='dialogue', sources=[source('D', 'final_score')]),
                                 layout, frame, sheet)
