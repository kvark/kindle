"""CPU protocol checks only: synthetic ledgers and real ALE, never a Kindle actor."""

import copy
from contextlib import ExitStack, closing
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import audit_atari_tasks as scorer
import replay_atari as replay


MINIMAL_ACTIONS = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']


def identity_fixture():
    rom = replay.rom_identity('ALE/Breakout-v5')
    native = Path(replay.ale_native.__file__)
    header = dict(environment='ALE/Breakout-v5', atari_protocol='published-minimal',
                  ale_py_version='0.12.1', wrapper_sha256=replay.sha256(replay.atari.__file__),
                  action_repeat=4, sticky_actions=0, noop_max=0, full_action_space=False,
                  max_episode_frames=100000, config=dict(action_count=4), action_meanings=MINIMAL_ACTIONS[:])
    manifest = dict(environment=header['environment'], atari_protocol='published-minimal',
                    action_count=4, action_meanings=MINIMAL_ACTIONS[:],
                    pins={rom['path']: rom['sha256'], str(native): replay.sha256(native)})
    return header, manifest, rom


def test_minimal_breakout_requires_a_new_replay_version_and_explicit_declaration():
    header, manifest, rom = identity_fixture()
    assert replay.verify_replay_identity(header, manifest, rom) == 'kindle-atari-task-replay-v2'
    del manifest['atari_protocol']
    with pytest.raises(ValueError, match='declared action protocol'):
        replay.verify_replay_identity(header, manifest, rom)


@pytest.mark.parametrize('declared', [False, True])
def test_published_replay_retains_v1_and_implicit_historical_declarations(declared):
    header, manifest, rom = identity_fixture()
    header.update(atari_protocol='published', full_action_space=True)
    if declared:
        manifest['atari_protocol'] = 'published'
    else:
        del manifest['atari_protocol']
    # The complete ledger and actual replay environment check the full action
    # vocabulary. V1 did not add a second action-count field to its manifest.
    for field in ('action_count', 'action_meanings'):
        del manifest[field]
    assert replay.verify_replay_identity(header, manifest, rom) == 'kindle-atari-task-replay-v1'


@pytest.mark.parametrize('mutation, message', [
    (lambda h, m: m.update(atari_protocol='published'), 'declared action protocol'),
    (lambda h, m: h.update(full_action_space=True), 'preprocessing'),
    (lambda h, m: h.update(full_action_space=0), 'preprocessing'),
    (lambda h, m: h.update(action_repeat=1), 'preprocessing'),
    (lambda h, m: h.update(sticky_actions=0.25), 'preprocessing'),
    (lambda h, m: h.update(noop_max=30), 'preprocessing'),
    (lambda h, m: h.update(max_episode_frames=108000), 'preprocessing'),
    (lambda h, m: h['config'].update(action_count=18), 'minimal action vocabulary'),
    (lambda h, m: h['config'].update(action_count=4.0), 'minimal action vocabulary'),
    (lambda h, m: h['config'].update(action_count=True), 'minimal action vocabulary'),
    (lambda h, m: h['config'].pop('action_count'), 'minimal action vocabulary'),
    (lambda h, m: h.update(action_meanings=['NOOP', 'FIRE', 'LEFT', 'RIGHT']), 'minimal action vocabulary'),
    (lambda h, m: m.update(action_count=18), 'declared minimal action vocabulary'),
    (lambda h, m: m.update(action_count=4.0), 'declared minimal action vocabulary'),
    (lambda h, m: m.update(action_count=True), 'declared minimal action vocabulary'),
    (lambda h, m: m.pop('action_count'), 'declared minimal action vocabulary'),
    (lambda h, m: m.pop('action_meanings'), 'declared minimal action vocabulary'),
    (lambda h, m: m.update(action_meanings=['NOOP', 'FIRE', 'LEFT', 'RIGHT']), 'declared minimal action vocabulary'),
    (lambda h, m: h.update(ale_py_version='other'), 'ALE version'),
    (lambda h, m: h.update(wrapper_sha256='wrong'), 'source wrapper'),
    (lambda h, m: m.update(pins={}), 'replay input'),
])
def test_minimal_protocol_does_not_accept_mixed_or_incomplete_inputs(mutation, message):
    header, manifest, rom = identity_fixture()
    mutation(header, manifest)
    with pytest.raises(ValueError, match=message):
        replay.verify_replay_identity(header, manifest, rom)


@pytest.mark.parametrize('environment', ['ALE/Freeway-v5', 'ALE/Qbert-v5', 'ALE/Pong-v5'])
def test_minimal_replay_is_not_implicitly_enabled_for_other_games(environment):
    header, manifest, rom = identity_fixture()
    header['environment'] = manifest['environment'] = environment
    with pytest.raises(ValueError, match='Breakout only'):
        replay.verify_replay_identity(header, manifest, rom)


def test_existing_current_preprocessing_is_not_a_published_minimal_replay():
    header, manifest, rom = identity_fixture()
    header['atari_protocol'] = manifest['atari_protocol'] = 'current'
    with pytest.raises(ValueError, match='unsupported replay preprocessing'):
        replay.verify_replay_identity(header, manifest, rom)


@pytest.mark.parametrize('minimal', [False, True])
def test_replay_cli_constructs_the_declared_real_ale_action_space(tmp_path, monkeypatch, minimal):
    """Replay a CPU-generated fixture; read_run is stubbed, not a native run certificate."""
    header, manifest, _ = identity_fixture()
    if not minimal:
        header.update(atari_protocol='published', full_action_space=True)
        del manifest['atari_protocol']
        del manifest['action_count']
        del manifest['action_meanings']
    header.update(num_envs=2, environment_seeds=[9001, 1009004], mode='evaluate_sample')
    replay.gym.register_envs(replay.ale_py)
    rows, episodes, returns, counts = [], [], [0.0, 0.0], [0, 0]
    with ExitStack() as stack:
        environments = []
        for seed in header['environment_seeds']:
            raw = stack.enter_context(closing(replay.gym.make(header['environment'], frameskip=1,
                repeat_action_probability=0.0, full_action_space=not minimal)))
            env = replay.atari.DreamerAtariPreprocessing(raw, noop_max=0, max_episode_frames=100000)
            env.reset(seed=seed)
            environments.append(env)
        header['action_meanings'] = list(environments[0].action_meanings)
        header['config']['action_count'] = len(header['action_meanings'])
        assert header['config']['action_count'] == (4 if minimal else 18)
        for tick in range(512):
            indices = [(tick + stream) % 4 for stream in range(2)]
            actions = [index if minimal else [0, 1, 3, 4][index] for index in indices]
            results = [env.step(action) for env, action in zip(environments, actions)]
            rows.append(dict(event='transition', run_step=(tick + 1) * 2, actions=actions,
                rewards=[r[1] for r in results], terminated=[r[2] for r in results],
                truncated=[r[3] for r in results],
                executed_action_frames=[env.executed_action_frames for env in environments]))
            resets = []
            for stream, result in enumerate(results):
                returns[stream] += result[1]
                if result[2] or result[3]:
                    row = dict(event='episode', stream=stream, episode=counts[stream],
                        episode_return=returns[stream], terminated=result[2], truncated=result[3])
                    rows.append(row)
                    episodes.append(copy.deepcopy(row))
                    counts[stream] += 1
                    returns[stream] = 0.0
                    resets.append(stream)
                    environments[stream].reset()
            if resets:
                rows.append(dict(event='reset', streams=resets))
        end = dict(executed_action_frames=[env.executed_action_frames for env in environments],
                   emulator_resets=[env.emulator_resets for env in environments], partial_returns=returns)
    assert episodes, 'fixture must exercise real episode boundaries and resets'
    log, declaration, output = tmp_path / 'fixture.jsonl', tmp_path / 'manifest.json', tmp_path / 'replay.json'
    log.write_text(''.join(json.dumps(row) + '\n' for row in [header] + rows))
    declaration.write_text(json.dumps(manifest))
    run = dict(path=str(log), sha256=replay.sha256(log), start=header,
               accounting=dict(updates=0, budget_complete=True), episodes=episodes, end=end)
    monkeypatch.setattr(replay, 'read_run', lambda path: run)
    monkeypatch.setattr(sys, 'argv', ['replay_atari.py', str(log), '--source-manifest', str(declaration),
                                    '--output', str(output)])
    replay.main()
    result = json.loads(output.read_text())
    assert result['protocol'] == f'kindle-atari-task-replay-v{2 if minimal else 1}'
    assert result['source_header']['action_meanings'] == header['action_meanings']
    scorer.check_replay(run, result)
    assert not scorer.score_tasks(header['environment'], result['episodes'])['task_gate_passed']
    result['protocol'] = f'kindle-atari-task-replay-v{1 if minimal else 2}'
    with pytest.raises(ValueError, match='unsupported replay protocol'):
        scorer.check_replay(run, result)
