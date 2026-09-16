"""Synthetic reader fixtures only; no native training or policy-quality claims."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import audit_atari_dose as a
from test_checkpoint_retention import case, freeway, retain, write_log
from test_atari_task_scores import replay_fixture


@pytest.fixture(params=['ALE/Qbert-v5', 'ALE/Freeway-v5'])
def pair(request):
    game = request.param
    start = {key: 'identical synthetic input' for key in a.SHARED}
    start.update(environment=game, config=dict(seed=0), num_envs=6)
    training = dict(start=start, accounting=dict(actions=200004), end=dict(unix_time=20))
    stage = dict(event='checkpoint', run_step=200004, learner_step=49651,
        identity=dict(metadata_sha256='synthetic metadata', tensor_sha256=dict(world='synthetic weight file')))
    header = dict(deepcopy(start), mode='evaluate_sample', num_envs=6, seed=a.EVALUATION_SEED,
        environment_seeds=[a.EVALUATION_SEED + stream * 1000003 for stream in range(6)],
        starting_environment_step=200004, starting_learner_step=49651, unix_time=30)
    if game == 'ALE/Qbert-v5':
        header.update(protocol=a.matches.EPISODE_EVALUATION_PROTOCOL, steps=600000, evaluation_episodes_per_stream=4)
        accounting = dict(actions=12300, updates=0, budget_complete=True,
            episode_budget_complete=True, evaluation_episodes_per_stream=4)
        end = dict(reason='episode_budget_complete', episode_counts=[4, 5, 6, 4, 5, 7], run_step=12300)
    else:
        header.update(protocol=a.retain.VECTOR_PROTOCOL, steps=75000)
        accounting = dict(actions=75000, updates=0, budget_complete=True)
        end = dict(reason='budget_complete', episode_counts=[6] * 6, run_step=75000)
    return training, dict(start=header, accounting=accounting, end=end), stage


def test_each_game_keeps_its_frozen_protocol(pair):
    a.check_frozen_protocol(*pair)


@pytest.mark.parametrize('field,value', [
    ('mode', 'evaluate_greedy'), ('protocol', 'unknown'), ('num_envs', 4), ('steps', 600006),
    ('seed', 100001), ('environment_seeds', list(range(6))), ('starting_environment_step', 400008),
    ('starting_learner_step', 1), ('unix_time', 0), ('native_extension_sha256', 'changed'),
    ('runner_sha256', 'changed'), ('config', {}), ('model_provenance', {}), ('gpu_device', {}),
    ('exploration', dict(probability=.5)), ('exploration_sha256', 'undeclared')])
def test_changed_frozen_protocol_rejected(pair, field, value):
    training, evaluation, stage = pair
    evaluation['start'][field] = value
    with pytest.raises((ValueError, KeyError)):
        a.check_frozen_protocol(training, evaluation, stage)


@pytest.mark.parametrize('field,value', [('updates', 1), ('budget_complete', False), ('actions', 600006), ('actions', 0)])
def test_wrong_accounting_rejected(pair, field, value):
    training, evaluation, stage = pair
    evaluation['accounting'][field] = value
    with pytest.raises(ValueError):
        a.check_frozen_protocol(training, evaluation, stage)


def test_games_cannot_exchange_evaluation_protocols(pair):
    training, evaluation, stage = pair
    if training['start']['environment'] == 'ALE/Qbert-v5':
        evaluation['start'].update(protocol=a.retain.VECTOR_PROTOCOL, steps=75000)
    else:
        evaluation['start'].update(protocol=a.matches.EPISODE_EVALUATION_PROTOCOL,
            steps=600000, evaluation_episodes_per_stream=4)
    with pytest.raises(ValueError):
        a.check_frozen_protocol(training, evaluation, stage)


def test_incomplete_boundary_is_not_a_result(pair):
    training, evaluation, stage = pair
    evaluation['end']['reason'] = 'action_cap_reached'
    with pytest.raises(ValueError):
        a.check_frozen_protocol(training, evaluation, stage)


def test_checkpoint_beyond_training_history_rejected(pair):
    training, evaluation, stage = pair
    stage['run_step'] = 400008
    with pytest.raises(ValueError, match='invalid checkpoint stage'):
        a.check_frozen_protocol(training, evaluation, stage)


@pytest.mark.parametrize('change', ['path', 'metadata', 'tensor'])
def test_restore_identity_must_be_the_saved_stage(pair, tmp_path, change):
    training, evaluation, stage = pair
    evaluation['start']['restored_checkpoint'] = dict(path=str(tmp_path), **deepcopy(stage['identity']))
    if change == 'path':
        evaluation['start']['restored_checkpoint']['path'] += '-different'
    elif change == 'metadata':
        evaluation['start']['restored_checkpoint']['metadata_sha256'] = 'other'
    else:
        evaluation['start']['restored_checkpoint']['tensor_sha256']['world'] = 'other'
    with pytest.raises(ValueError, match='undeclared restored|not the declared stage'):
        a.check_saved_stage(training, stage, evaluation, tmp_path, 'not reached', 'not reached')


@pytest.mark.parametrize('game', ['case', 'freeway'])
def test_retained_component_is_bound_to_original_prefix(request, game):
    fixture = request.getfixturevalue(game)
    retain(fixture)
    training = dict(path=str(fixture['log']), start=fixture['rows'][0])
    record, event, checkpoint = a.check_retained_stage(training, fixture['destination'], fixture['action'])
    assert event == fixture['rows'][-1] and checkpoint == fixture['destination'] / 'checkpoint'
    assert not record['complete_training']


def test_retained_prefix_is_not_complete_training(case):
    retain(case)
    with pytest.raises(ValueError, match='missing run_end'):
        a.read_training(case['destination'] / 'training-prefix.jsonl', case['expected_header'], 12, 'not reached')


@pytest.mark.parametrize('change', ['protocol', 'hash', 'event', 'path', 'complete'])
def test_changed_archive_binding_rejected(case, change):
    record = retain(case)
    if change == 'protocol':
        record['protocol'] = 'kindle-qbert-settled-checkpoint-retention-v1'
    elif change == 'hash':
        record['source_prefix_sha256'] = '0' * 64
    elif change == 'event':
        record['source_checkpoint_event']['learner_step'] = 999
    elif change == 'path':
        record['archived_checkpoint'] += '-different'
    else:
        record['complete_training'] = True
    (case['destination'] / 'retained.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):
        a.check_retained_stage(dict(path=str(case['log']), start=case['rows'][0]), case['destination'], 12)


def synthetic_scores():
    stages = [dict(checkpoint=dict(checkpoint_stage_actions=action, checkpoint_stage_updates=updates),
        score=dict(task_gate_passed=passed, mean_completed_return=mean), full_training_sha256='same synthetic history')
        for action, updates, passed, mean in ((200004, 1, False, 4000), (400008, 2, True, 16000))]
    control = dict(score=dict(task_gate_passed=False, mean_completed_return=125))
    return stages, control


def test_one_successful_pair_is_not_reliability():
    result = a.summarize(*synthetic_scores())
    assert result['final_paired_gate_passed'] and result['final_minus_midpoint_mean'] == 12000
    assert result['independent_training_roots'] == 1
    assert not result['reliability_assessed'] and not result['five_game_goal_complete']


@pytest.mark.parametrize('change', ['missing_midpoint', 'reversed', 'same_update', 'different_histories',
    'failed_final', 'passing_control', 'inferior_final', 'empty_control'])
def test_invalid_or_failed_pair_does_not_pass(change):
    stages, control = synthetic_scores()
    if change == 'missing_midpoint':
        stages = stages[1:]
    elif change == 'reversed':
        stages.reverse()
    elif change == 'same_update':
        stages[1]['checkpoint']['checkpoint_stage_updates'] = 1
    elif change == 'different_histories':
        stages[1]['full_training_sha256'] = 'other synthetic history'
    elif change == 'failed_final':
        stages[1]['score']['task_gate_passed'] = False
    elif change == 'passing_control':
        control['score']['task_gate_passed'] = True
    elif change == 'inferior_final':
        control['score']['mean_completed_return'] = 17000
    else:
        control['score']['mean_completed_return'] = None
    if change in ('missing_midpoint', 'reversed', 'same_update', 'different_histories'):
        with pytest.raises(ValueError):
            a.summarize(stages, control)
    else:
        assert not a.summarize(stages, control)['final_paired_gate_passed']


@pytest.fixture
def declaration(case, tmp_path):
    source = tmp_path / 'synthetic-source'
    source.mkdir()
    for name in ('atari_vector.py', 'atari.py'):
        (source / name).write_text('# Synthetic identity fixture; not an executable agent.\n')
    native = tmp_path / 'synthetic-native-identity'
    native.write_bytes(b'Explicit fixture, not a loadable native extension.')
    expected = deepcopy(case['expected_header'])
    expected.update(steps=400008, config=dict(a.RECIPE, seed=0,
        loss_scales=dict(reconstruction=0, future_prediction=0.25)),
        native_extension_sha256=a.digest(native), runner_sha256=a.digest(source / 'atari_vector.py'),
        wrapper_sha256=a.digest(source / 'atari.py'))
    metadata = a.read(case['schema'] / 'metadata.json')
    metadata['config'] = deepcopy(expected['config'])
    (case['schema'] / 'metadata.json').write_text(json.dumps(metadata))
    paths = [Path(a.__file__), Path(a.retain.__file__), Path(a.matches.__file__), Path(a.tasks.__file__),
        source / 'atari_vector.py', source / 'atari.py', native, case['encoder'],
        *(case['schema'] / name for name in a.retain.FILES)]
    return dict(protocol='kindle-atari-continuous-dose-v1', training_actions=400008,
        stages=[dict(actions=action) for action in a.STAGES], training_header=expected,
        criteria=deepcopy(a.tasks.TASK_CRITERIA[expected['environment']]), source=str(source),
        schema=str(case['schema']), encoder=str(case['encoder']), native_extension=str(native),
        pins={str(path.resolve()): a.digest(path) for path in paths})


def test_declaration_binds_recipe_sources_and_state_schema(declaration):
    a.verify_declaration(declaration)


@pytest.mark.parametrize('change', ['budget', 'stage', 'header_budget', 'ratio', 'criteria',
    'provenance', 'pin', 'native', 'runner', 'wrapper'])
def test_changed_declared_study_is_rejected(declaration, change):
    if change == 'budget':
        declaration['training_actions'] = 200004
    elif change == 'stage':
        declaration['stages'].pop()
    elif change == 'header_budget':
        declaration['training_header']['steps'] = 200004
    elif change == 'ratio':
        declaration['training_header']['config']['train_ratio'] = 64
    elif change == 'criteria':
        declaration['criteria']['minimum_mean_return'] = 1000
    elif change == 'provenance':
        declaration['training_header']['model_provenance']['meganeura_revision'] = 'other'
    elif change == 'pin':
        declaration['pins'].pop(declaration['native_extension'])
    else:
        key = dict(native='native_extension_sha256', runner='runner_sha256', wrapper='wrapper_sha256')[change]
        declaration['training_header'][key] = '0' * 64
    with pytest.raises(ValueError):
        a.verify_declaration(declaration)


def test_rehashed_reference_cannot_lower_the_fixed_replay_ratio(declaration):
    path = Path(declaration['schema']) / 'metadata.json'
    metadata = a.read(path)
    metadata['config']['train_ratio'] = 64
    path.write_text(json.dumps(metadata))
    declaration['pins'][str(path)] = a.digest(path)
    declaration['training_header']['config']['train_ratio'] = 64
    with pytest.raises(ValueError, match='fixed dose recipe'):
        a.verify_declaration(declaration)


def test_changed_pinned_input_is_rejected(declaration):
    Path(declaration['native_extension']).write_bytes(b'changed synthetic fixture')
    with pytest.raises(ValueError, match='changed input'):
        a.verify_declaration(declaration)


def test_video_includes_partial_tail_but_score_does_not(tmp_path):
    evaluation, replay = replay_fixture(tmp_path)
    video = tmp_path / 'synthetic-video-identity.txt'
    video.write_text('File identity fixture, not an encoded rollout video.\n')
    replay['video'] = dict(stream=0, fps=60, frames=1000, path=str(video), sha256=a.digest(video))
    path = tmp_path / 'replay.json'
    path.write_text(json.dumps(replay))
    result = a.score_replayed(evaluation, path)
    assert result['score']['mean_completed_return'] == 3950
    assert not result['score']['task_gate_passed']
    assert result['replay']['partial'][1]['episode_score'] == 50000


@pytest.mark.parametrize('field,value', [('stream', 1), ('fps', 30), ('frames', 999), ('sha256', 'wrong')])
def test_changed_video_binding_fails(tmp_path, field, value):
    evaluation, replay = replay_fixture(tmp_path)
    video = tmp_path / 'synthetic-video-identity.txt'
    video.write_text('File identity fixture, not an encoded rollout video.\n')
    replay['video'] = dict(stream=0, fps=60, frames=1000, path=str(video), sha256=a.digest(video))
    replay['video'][field] = value
    path = tmp_path / 'replay.json'
    path.write_text(json.dumps(replay))
    with pytest.raises(ValueError, match='whole-stream video'):
        a.score_replayed(evaluation, path)


def test_cli_refuses_existing_output_before_reading_a_declaration(tmp_path):
    path = tmp_path / 'existing.json'
    path.write_text('keep this file\n')
    with pytest.raises(ValueError, match='fresh'):
        a.main(['--declaration', 'not read', '--output', str(path)])
    assert path.read_text() == 'keep this file\n'
