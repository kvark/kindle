from copy import deepcopy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import retain_atari_checkpoint as r


def identity(path):
    return dict(metadata_sha256=r.digest(path / 'metadata.json'),
                tensor_sha256={name: r.digest(path / f'{name}.safetensors') for name in r.PARTS})


def write_log(case):
    case['log'].write_text(''.join(json.dumps(row) + '\n' for row in case['rows']))


def rebind(case):
    metadata = r.read(case['checkpoint'] / 'metadata.json')
    metadata['tensor_sha256'] = {name: r.digest(case['checkpoint'] / f'{name}.safetensors') for name in r.PARTS}
    (case['checkpoint'] / 'metadata.json').write_text(json.dumps(metadata))
    case['rows'][-1]['identity'] = identity(case['checkpoint'])
    write_log(case)


@pytest.fixture
def case(tmp_path):
    encoder = tmp_path / 'encoder'
    encoder.write_bytes(b'synthetic encoder identity, not a usable model')
    perception = dict(kind='levjepa', checkpoint_sha256=r.digest(encoder))
    provenance = dict(dreamerv3_revision='d3', meganeura_revision='mega', blade_revision='blade',
                      future_head_revision='future', perception=perception)
    config = dict(batch_size=1, batch_length=2, replay_context=1, replay_capacity=64, action_count=18, train_ratio=2.0)
    header = dict(event='run_start', protocol=r.VECTOR_PROTOCOL, environment='ALE/Qbert-v5', num_envs=6,
        steps=24, seed=0, mode='train', environment_seeds=list(range(6)), config=config, model_provenance=provenance,
        restored_checkpoint=None, starting_environment_step=0, starting_learner_step=0, action_repeat=4)
    rows = [header]
    for tick in (1, 2):
        rows.append(dict(event='transition', run_step=tick * 6, vector_tick=tick, actions=[2] * 6,
            rewards=[0.0] * 6, stored_rewards=[[0.0, 0.0] for _ in range(6)], terminated=[False] * 6,
            truncated=[False] * 6, executed_action_frames=[tick * 4] * 6))
    rows.append(dict(event='learner', run_step=12, report=dict(learner_step=1, replay_len=18, loss=1.0)))
    metadata = dict(format=3, architecture='dreamerv3-visual-features', config=config, **provenance,
        environment_step=12, learner_step=1, collection_streams=6, return_low=0.0, return_high=1.0)
    checkpoint, schema = tmp_path / 'source', tmp_path / 'schema'
    for directory in (checkpoint, schema):
        directory.mkdir()
        for name, count, adam_count in (('world', 62, 51), ('behavior', 22, 22), ('slow_value', 11, 0)):
            parameters = {f'{name}.p{index}': dict(shape=[1], dtype='F32') for index in range(count)}
            adam = list(parameters)[:adam_count]
            values = {key: np.ones(1, dtype=np.float32) for key in parameters}
            values.update({f'{kind}.{key}': np.full(1, .25, dtype=np.float32)
                           for kind in ('adam_m', 'adam_v') for key in adam})
            save_file(values, str(directory / f'{name}.safetensors'), metadata=dict(
                meganeura_checkpoint_format='3', adam_step='0' if name == 'slow_value' else '1',
                meganeura_logical_layout=json.dumps(dict(parameters=parameters, adam_parameters=adam))))
        current = dict(metadata, tensor_sha256={name: r.digest(directory / f'{name}.safetensors') for name in r.PARTS})
        (directory / 'metadata.json').write_text(json.dumps(current))
    rows.append(dict(event='checkpoint', run_step=12, learner_step=1, identity=identity(checkpoint)))
    result = dict(log=tmp_path / 'source.jsonl', checkpoint=checkpoint, schema=schema, encoder=encoder,
        destination=tmp_path / 'retained', rows=rows, expected_header=deepcopy(header), action=12)
    write_log(result)
    return result


def retain(case):
    return r.retain(**{key: value for key, value in case.items() if key != 'rows'})


def state(case):
    return r.verify_state(case['checkpoint'], case['rows'][-1]['identity'], case['rows'][0],
        case['rows'][-1], case['schema'], case['encoder'])


def rewrite_tensor(case, change):
    path = case['checkpoint'] / 'world.safetensors'
    with safe_open(path, framework='numpy') as model:
        values, metadata = {key: model.get_tensor(key) for key in model.keys()}, model.metadata()
    change(values, metadata)
    save_file(values, str(path), metadata=metadata)
    rebind(case)


def test_retains_complete_prefix_state_without_source_writes(case):
    paths = [case['log'], *(case['checkpoint'] / name for name in r.FILES)]
    before = {path: r.digest(path) for path in paths}
    result = retain(case)
    assert result['complete_finite_state'] == dict(tensors=r.COUNTS, parameters=95, optimizer_moments=146,
        environment_step=12, learner_step=1)
    assert result['prefix_expected_error'] == 'missing run_end'
    assert not result['complete_training'] and not result['learner_called'] and not result['checkpoint_restored']
    assert all(r.digest(path) == value for path, value in before.items())
    assert r.read(case['destination'] / 'retained.json') == result


def test_later_log_growth_is_not_a_restart_or_a_second_prefix(case):
    prefix = case['log'].read_bytes()
    with case['log'].open('a') as output:
        output.write(json.dumps(dict(event='transition', run_step=18)) + '\n')
    retain(case)
    assert (case['destination'] / 'training-prefix.jsonl').read_bytes() == prefix


def test_existing_destination_is_never_reused(case):
    case['destination'].mkdir()
    with pytest.raises(ValueError, match='destination must be fresh'):
        retain(case)


def test_source_overlap_is_rejected(case):
    case['destination'] = case['checkpoint'] / 'nested'
    with pytest.raises(ValueError, match='overlaps'):
        retain(case)


@pytest.mark.parametrize('field,value', [('seed', 1), ('steps', 30), ('num_envs', 8), ('model_provenance', {})])
def test_changed_declared_header_rejected(case, field, value):
    case['expected_header'][field] = value
    with pytest.raises(ValueError, match='declared source header changed'):
        retain(case)


@pytest.mark.parametrize('action', [0, -6, 5, 12.0, True, 18])
def test_invalid_or_missing_checkpoint_rejected(case, action):
    case['action'] = action
    with pytest.raises(ValueError):
        retain(case)
    assert not case['destination'].exists()


def test_partial_log_row_does_not_authorize_capture(case):
    case['log'].write_bytes(case['log'].read_bytes()[:-1])
    with pytest.raises(ValueError, match='unfinished source row'):
        retain(case)


def test_wrong_checkpoint_counter_is_rejected_by_complete_ledger(case):
    case['rows'][-1]['learner_step'] = 2
    write_log(case)
    with pytest.raises(ValueError, match='checkpoint counters mismatch'):
        r.checked_prefix(case['log'])


def test_missing_learner_row_is_not_accepted_as_incomplete_end(case):
    del case['rows'][-2]
    write_log(case)
    with pytest.raises(ValueError, match='incomplete vector round'):
        r.checked_prefix(case['log'])


@pytest.mark.parametrize('field,value', [('format', 2), ('architecture', 'other'), ('environment_step', 18),
    ('learner_step', 2), ('collection_streams', 8), ('return_low', float('nan')), ('return_high', -1.0)])
def test_wrong_top_level_saved_state_rejected(case, field, value):
    path = case['checkpoint'] / 'metadata.json'
    metadata = r.read(path)
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    rebind(case)
    with pytest.raises(ValueError):
        state(case)


def test_actual_encoder_is_required(case):
    case['encoder'].write_bytes(b'different file with the same hypothetical tensor shapes')
    with pytest.raises(ValueError, match='actual encoder identity'):
        state(case)


@pytest.mark.parametrize('kind', ['missing_parameter', 'missing_moment', 'shape', 'dtype', 'nan', 'negative_v', 'step', 'layout'])
def test_corrupt_logical_weights_or_moments_rejected_even_with_rehashed_files(case, kind):
    def change(values, metadata):
        if kind == 'missing_parameter':
            values.pop('world.p0')
        elif kind == 'missing_moment':
            values.pop('adam_m.world.p0')
        elif kind == 'shape':
            values['world.p0'] = np.ones(2, dtype=np.float32)
        elif kind == 'dtype':
            values['world.p0'] = np.ones(1, dtype=np.float64)
        elif kind == 'nan':
            values['world.p0'][0] = np.nan
        elif kind == 'negative_v':
            values['adam_v.world.p0'][0] = -1
        elif kind == 'step':
            metadata['adam_step'] = '2'
        else:
            layout = json.loads(metadata['meganeura_logical_layout'])
            layout['parameters']['world.p0']['shape'] = [2]
            metadata['meganeura_logical_layout'] = json.dumps(layout)
    rewrite_tensor(case, change)
    with pytest.raises(ValueError):
        state(case)


def test_changed_file_is_detected_before_copy(case):
    with (case['checkpoint'] / 'world.safetensors').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checkpoint file identity'):
        retain(case)
    assert not case['destination'].exists()


def test_source_change_during_copy_leaves_no_completion_marker(case, monkeypatch):
    original = r.copy_exclusive
    def changed(source, destination, limit=None):
        original(source, destination, limit)
        if Path(source).name == 'world.safetensors':
            with Path(source).open('ab') as stream:
                stream.write(b'changed during copy')
    monkeypatch.setattr(r, 'copy_exclusive', changed)
    with pytest.raises(ValueError, match='checkpoint file identity'):
        retain(case)
    assert case['destination'].is_dir()
    assert not (case['destination'] / 'retained.json').exists()
    with pytest.raises(ValueError, match='destination must be fresh'):
        retain(case)


def test_copy_refuses_preexisting_output(case, tmp_path):
    target = tmp_path / 'existing'
    target.write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        r.copy_exclusive(case['log'], target)
    assert target.read_bytes() == b'keep'


@pytest.fixture
def freeway(case):
    import kindle._exploration as exploration
    header = case['rows'][0]
    header.update(environment='ALE/Freeway-v5', protocol=exploration.EXPLORATION_PROTOCOL,
        exploration=dict(kind=exploration.EXPLORATION_KIND, probability=0.5, hold_actions=64, seed=0),
        exploration_sha256=r.digest(exploration.__file__))
    header['config']['seed'] = 0
    policy = exploration.PersistentExploration(header['exploration'], 6, 18)
    for row in case['rows']:
        if row['event'] == 'transition':
            row['action_overrides'] = policy.actions()
            row['actions'] = [2 if forced is None else forced for forced in row['action_overrides']]
    for directory in (case['checkpoint'], case['schema']):
        path = directory / 'metadata.json'
        metadata = r.read(path)
        metadata['config']['seed'] = 0
        path.write_text(json.dumps(metadata))
    case['expected_header'] = deepcopy(header)
    rebind(case)
    return case


def test_freeway_retains_exact_override_ledger_and_all_moments(freeway):
    result = retain(freeway)
    assert result['protocol'] == 'kindle-atari-settled-checkpoint-retention-v1'
    assert result['source_header']['environment'] == 'ALE/Freeway-v5'
    assert result['complete_finite_state']['optimizer_moments'] == 146
    assert (freeway['destination'] / 'training-prefix.jsonl').read_bytes() == freeway['log'].read_bytes()


@pytest.mark.parametrize('field,value', [('probability', 0.25), ('hold_actions', 1),
                                       ('kind', 'other'), ('seed', 1)])
def test_freeway_does_not_change_exploration_with_experience_budget(freeway, field, value):
    freeway['rows'][0]['exploration'][field] = value
    freeway['expected_header'] = deepcopy(freeway['rows'][0])
    write_log(freeway)
    with pytest.raises(ValueError, match='Freeway must retain'):
        retain(freeway)
    assert not freeway['destination'].exists()


def test_unassisted_freeway_is_not_this_dose_comparison(case):
    case['rows'][0]['environment'] = 'ALE/Freeway-v5'
    case['expected_header'] = deepcopy(case['rows'][0])
    write_log(case)
    with pytest.raises(ValueError, match='Freeway must retain'):
        retain(case)


def test_qbert_cannot_acquire_freeway_assistance(freeway):
    freeway['rows'][0]['environment'] = 'ALE/Qbert-v5'
    freeway['expected_header'] = deepcopy(freeway['rows'][0])
    write_log(freeway)
    with pytest.raises(ValueError, match='Qbert must remain unassisted'):
        retain(freeway)


@pytest.mark.parametrize('change', ['override', 'action', 'implementation', 'seed'])
def test_freeway_rejects_broken_exploration_even_with_matching_header(freeway, change):
    header = freeway['rows'][0]
    first = freeway['rows'][1]
    if change == 'override':
        first['action_overrides'] = [18] * 6
    elif change == 'action':
        stream = next(i for i, forced in enumerate(first['action_overrides']) if forced is not None)
        first['actions'][stream] = (first['actions'][stream] + 1) % 18
    elif change == 'implementation':
        header['exploration_sha256'] = 'wrong'
    else:
        header['config']['seed'] = 1
        for directory in (freeway['checkpoint'], freeway['schema']):
            path = directory / 'metadata.json'
            metadata = r.read(path)
            metadata['config']['seed'] = 1
            path.write_text(json.dumps(metadata))
        rebind(freeway)
    freeway['expected_header'] = deepcopy(header)
    write_log(freeway)
    with pytest.raises(ValueError, match='prefix audit failed'):
        retain(freeway)
    assert not (freeway['destination'] / 'retained.json').exists()


@pytest.mark.parametrize('environment', ['ALE/Pong-v5', 'ALE/Boxing-v5', 'ALE/Breakout-v5'])
def test_other_games_require_their_own_retention_contract(case, environment):
    case['rows'][0]['environment'] = environment
    case['expected_header'] = deepcopy(case['rows'][0])
    write_log(case)
    with pytest.raises(ValueError, match='not fresh vector'):
        retain(case)


def test_cli_copies_a_declared_snapshot_and_refuses_reuse(freeway, tmp_path, capsys):
    header = tmp_path / 'header.json'
    header.write_text(json.dumps(freeway['expected_header']))
    args = [value for name in ('log', 'checkpoint', 'destination', 'schema', 'encoder')
            for value in ('--' + name, str(freeway[name]))]
    args += ['--header', str(header), '--action', str(freeway['action'])]
    r.main(args)
    assert json.loads(capsys.readouterr().out)['learner_called'] is False
    with pytest.raises(ValueError, match='destination must be fresh'):
        r.main(args)
