import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import kindle
import kindle._qbert_guidance as module
from kindle._qbert_guidance import BOUNDED_GUIDANCE_KIND, GUIDANCE_KIND, GUIDANCE_SEED_XOR, QbertGuidance
from kindle._exploration import EXPLORATION_PROTOCOL
from kindle._vector_audit import audit
from test_vector import version_two_events, write_log
import atari_vector
import replay_qbert_guidance


def config(probability=.05):
    return dict(kind=GUIDANCE_KIND, probability=probability, seed=1009)


@pytest.mark.parametrize('field,value', [('probability', 0), ('probability', -1), ('probability', 1.1),
    ('probability', float('nan')), ('probability', float('inf')), ('probability', True),
    ('seed', -1), ('seed', 2**64), ('seed', True), ('kind', 'unknown')])
def test_reject_invalid_config(field, value):
    spec = config(); spec[field] = value
    with pytest.raises(ValueError):
        QbertGuidance(spec, 8, 18)


def test_schedule_draws_once_per_episode_and_lanes_are_independent():
    guided = QbertGuidance(config(.3), 3, 18)
    rngs = [random.Random((1009 + i) ^ GUIDANCE_SEED_XOR) for i in range(3)]
    expected = [rng.random() < .3 for rng in rngs]
    counts = [0] * 3
    for tick in range(200):
        overrides = [0 if active else None for active in expected]
        guided.verify_actions(overrides)
        counts = [n + active for n, active in zip(counts, expected)]
        assert guided.overridden_actions == counts
        lane = tick % 3
        if tick % 7 == 0:
            guided.reset([lane])
            expected[lane] = rngs[lane].random() < .3
    assert sum(guided.episode_counts['started_episodes']) == 32


def test_unguided_lanes_do_not_inspect_frames_and_reset_clears_only_one_policy():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('unguided pixels inspected')
    guided = QbertGuidance(config(1e-20), 2, 18)
    assert guided.actions([Poison(), Poison()]) == [None, None]
    assert guided.frame_sha256 == [None, None]
    all_guided = QbertGuidance(config(1), 2, 18)
    first, second = all_guided._policies
    first.lit[(0, 0)] = second.lit[(0, 0)] = True
    all_guided.reset([0])
    assert all_guided._policies[0] is not first and not all_guided._policies[0].lit[(0, 0)]
    assert all_guided._policies[1] is second and second.lit[(0, 0)]
    with pytest.raises(ValueError):
        all_guided.reset([0, 0])


def guided_events(spec=None):
    spec = config(1) if spec is None else spec
    rows = version_two_events()
    rows[0].update(protocol=EXPLORATION_PROTOCOL, seed=1009, environment='ALE/Qbert-v5',
        atari_protocol='published', full_action_space=True, observation_size='native',
        observation_shape=[210, 160, 3], sticky_actions=.25, noop_max=0, action_repeat=4,
        exploration=spec, exploration_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    rows[0]['config'].update(seed=1009, action_count=18)
    guided = QbertGuidance(spec, 2, 18)
    frames = [np.zeros((210,160,3), dtype=np.uint8)] * 2
    for row in rows[1:]:
        if row['event'] == 'transition':
            row['action_overrides'] = guided.actions(frames)
            row['actions'] = [1 if forced is None else forced for forced in row['action_overrides']]
            row['guidance_frame_sha256'] = guided.frame_sha256
        elif row['event'] == 'reset':
            guided.reset(row['streams'])
        elif row['event'] in ('progress', 'run_end'):
            row['overridden_actions'] = guided.overridden_actions
            row['guidance_episodes'] = guided.episode_counts
    return rows


def test_accounting_distinguishes_schedule_from_pixel_verification(tmp_path):
    result = audit(write_log(tmp_path, guided_events()))
    assert result['actions'] == 6 and result['updates'] == 3
    assert result['overridden_actions'] == [3, 3]
    assert result['exploration_schedule_verified'] is True
    assert result['exploration_pixel_actions_verified'] is False
    assert result['exploration_ledger_verified'] is False


def test_training_replay_reader_accepts_cdp_intrinsic_rewards(tmp_path):
    rows = guided_events()
    rows[0]['config'].update(disagreement_bonus=True, intrinsic_reward_scale=1.0,
                             extrinsic_reward_scale=1.0, visitation_bonus=False)
    result = replay_qbert_guidance.read_guided_run(write_log(tmp_path, rows))
    assert result['accounting']['actions'] == 6 and result['accounting']['updates'] == 3
    assert result['start']['config']['intrinsic_reward_scale'] == 1.0


@pytest.mark.parametrize('mutation', ['frozen', 'wrong_game', 'hash', 'override', 'input_hash', 'counts', 'episodes'])
def test_accounting_rejects_bad_guidance(tmp_path, mutation):
    rows = guided_events(); transition = next(r for r in rows if r['event'] == 'transition')
    if mutation == 'frozen': rows[0]['mode'] = 'evaluate_sample'
    elif mutation == 'wrong_game': rows[0]['environment'] = 'ALE/Pong-v5'
    elif mutation == 'hash': rows[0]['exploration_sha256'] = 'bad'
    elif mutation == 'override': transition['action_overrides'] = [None, None]
    elif mutation == 'input_hash': transition['guidance_frame_sha256'][0] = None
    elif mutation == 'counts': rows[-1]['overridden_actions'][0] += 1
    else: rows[-1]['guidance_episodes']['guided_episodes'][0] += 1
    with pytest.raises(ValueError): audit(write_log(tmp_path, rows))


@pytest.mark.parametrize('args,message', [
    (['--qbert-guidance-probability', 'nan'], 'probability'),
    (['--qbert-guidance-probability', '-.1'], 'probability'),
    (['--qbert-guidance-probability', '1.1'], 'probability'),
    (['--evaluate'], 'frozen evaluation'), (['--restore', 'unused'], 'fresh run'),
    (['--exploration-probability', '.05'], 'cannot combine'),
    (['--observation-size', '64'], 'native pixels'),
    (['--atari-protocol', 'published-minimal'], 'full18'),
    (['--encoder-checkpoint', 'unused'], 'learned RGB'),
    (['--qbert-guidance-max-actions', '0'], 'positive u32'),
    (['--qbert-guidance-max-actions', '-1'], 'positive u32'),
    (['--qbert-guidance-max-actions', str(2**32)], 'positive u32'),
    (['--qbert-guidance-max-actions', '1.5'], 'invalid int'),
    (['--qbert-guidance-max-actions', '640', '--qbert-guidance-probability', '0'], 'requires positive'),
    (['--qbert-guidance-max-actions', '640', '--evaluate'], 'frozen evaluation'),
    (['--qbert-guidance-max-actions', '640', '--restore', 'unused'], 'fresh run'),
])
def test_cli_refuses_invalid_guidance_before_gpu(monkeypatch, tmp_path, capsys, args, message):
    monkeypatch.setattr(atari_vector.gym, 'make', lambda *a, **k: pytest.fail('environment constructed'))
    monkeypatch.setattr(sys, 'argv', ['atari_vector', 'ALE/Qbert-v5', '--output', str(tmp_path/'run'),
                                     '--qbert-guidance-probability', '.05', *args])
    with pytest.raises(SystemExit) as error: atari_vector.main()
    assert error.value.code == 2 and message in capsys.readouterr().err


@pytest.mark.parametrize('cap', [None, 1])
def test_runner_uses_current_pixels_and_reset_pixels_and_native_overrides(monkeypatch, tmp_path, cap):
    created = []
    class Policy:
        def action(self, frame): return int(frame[0,0,0])
    monkeypatch.setattr(module, 'PixelPolicy', Policy)
    class Environment:
        action_space = SimpleNamespace(n=18)
        action_meanings = [str(n) for n in range(18)]
        executed_action_frames = reset_noop_frames = emulator_resets = 0
        def reset(self, seed=None):
            self.length = 0; self.emulator_resets += 1
            return np.full((210,160,3), 0 if seed is not None else 3, dtype=np.uint8), {}
        def step(self, action):
            self.actions.append(action); self.length += 1; self.executed_action_frames += 4
            return np.full((210,160,3), 2 if self.length == 1 else 4, dtype=np.uint8), 1., self.length == 2, False, {}
        def close(self): pass
    class Agent:
        environment_step = learner_step = replay_len = 0
        training_debt = 0.
        provenance = gpu_device = {}
        cpu_worker_threads = 1
        trainable_parameter_counts = dict(world=0, behavior=0)
        @classmethod
        def learned_rgb(cls, streams, config):
            obj = cls(); obj.config = config; return obj
        def begin_episodes(self, ids, frames): self.replay_len += len(ids)
        def act(self, greedy=False, action_overrides=None):
            return [17 if forced is None else forced for forced in action_overrides]
        def observe(self, ids, frames, rewards, terminated, truncated):
            self.environment_step += len(ids); self.replay_len += len(ids)
            return [[reward, 0.] for reward in rewards]
        def learn_scheduled(self): return []
    def make(*a, **kw):
        env = Environment(); env.actions = []; created.append(env); return env
    monkeypatch.setattr(atari_vector.gym, 'make', make)
    monkeypatch.setattr(atari_vector, 'DreamerAtariPreprocessing', lambda env, **kw: env)
    monkeypatch.setattr(kindle, 'VectorAgent', Agent)
    output = tmp_path/'run.jsonl'
    monkeypatch.setattr(sys, 'argv', ['atari_vector', 'ALE/Qbert-v5', '--output', str(output),
        '--num-envs', '2', '--steps', '6', '--train-ratio', '0', '--qbert-guidance-probability', '1',
        *([] if cap is None else ['--qbert-guidance-max-actions', str(cap)])])
    atari_vector.main()
    assert [env.actions for env in created] == ([[0,2,3], [0,2,3]] if cap is None else [[0,17,3], [0,17,3]])
    result = audit(output)
    assert result['actions'] == 6 and result['updates'] == 0
    assert result['overridden_actions'] == ([3,3] if cap is None else [2,2])
    assert result['guidance_episodes'] == dict(started_episodes=[2,2], guided_episodes=[2,2])


@pytest.mark.parametrize('mutation', ['action', 'pixel', 'execution'])
def test_pixel_replay_rejects_forged_teacher_choices_before_environment_step(mutation):
    frame = np.zeros((210,160,3), dtype=np.uint8)
    guided = QbertGuidance(config(1), 1, 18)
    row = dict(event='transition', action_overrides=[0], guidance_frame_sha256=[hashlib.sha256(frame.tobytes()).hexdigest()], actions=[0])
    if mutation == 'action': row['action_overrides'] = [2]
    elif mutation == 'pixel': row['guidance_frame_sha256'] = ['0'*64]
    else: row['actions'] = [2]
    with pytest.raises(ValueError):
        replay_qbert_guidance.replay_rows([json.dumps(row)], [], [], [frame], guided, float('inf'))


def bounded_config(cap=2):
    return dict(config(1), kind=BOUNDED_GUIDANCE_KIND, max_actions=cap)


@pytest.mark.parametrize('cap', [None, 0, -1, 2**32, True, 1.5, float('nan')])
def test_bounded_guidance_requires_positive_integer_limit(cap):
    with pytest.raises(ValueError, match='positive u32'):
        QbertGuidance(bounded_config(cap), 2, 18)


def test_whole_episode_kind_rejects_silent_limit():
    with pytest.raises(ValueError, match='fields'):
        QbertGuidance(dict(config(1), max_actions=640), 2, 18)


def test_cap_releases_without_redraw_or_pixel_access_and_lane_reset_is_independent():
    class Poison:
        def __getattribute__(self, name): raise AssertionError('released pixels inspected')
    frame = np.zeros((210,160,3), dtype=np.uint8)
    guided = QbertGuidance(bounded_config(), 2, 18)
    assert guided.actions([frame,frame]) == [0,0]
    assert guided.actions([frame,frame]) == [0,0]
    assert guided.actions([Poison(),Poison()]) == [None,None]
    assert guided.frame_sha256 == [None,None]
    assert guided.selected_episodes == [True,True]
    assert guided.episode_counts == dict(started_episodes=[1,1], guided_episodes=[1,1])
    assert guided.overridden_actions == [2,2]
    guided.reset([0])
    assert guided.actions([frame,Poison()]) == [0,None]
    assert guided.overridden_actions == [3,2]
    assert guided.episode_counts == dict(started_episodes=[2,1], guided_episodes=[2,1])


@pytest.mark.parametrize('mutation', ['early_release','late_override','hash_after_release','limit'])
def test_cap_accounting_rejects_forged_release(tmp_path, mutation):
    rows = guided_events(bounded_config(1))
    transitions = [r for r in rows if r['event']=='transition']
    if mutation=='early_release': transitions[0]['action_overrides'][0]=None
    elif mutation=='late_override': transitions[1]['action_overrides'][0]=0
    elif mutation=='hash_after_release': transitions[1]['guidance_frame_sha256'][0]='0'*64
    else: rows[0]['exploration']['max_actions']=2
    with pytest.raises(ValueError): audit(write_log(tmp_path,rows))


def test_cap_accounting_keeps_learning_and_episode_selection_after_release(tmp_path):
    rows=guided_events(bounded_config(1))
    result=replay_qbert_guidance.read_guided_run(write_log(tmp_path,rows))
    assert result['accounting']['actions']==6 and result['accounting']['updates']==3
    assert result['accounting']['overridden_actions']==[2,1]
    assert result['accounting']['guidance_episodes']==dict(started_episodes=[2,1],guided_episodes=[2,1])


def test_replay_preserves_selected_episode_label_after_guidance_release():
    frame=np.zeros((210,160,3),dtype=np.uint8)
    digest=hashlib.sha256(frame.tobytes()).hexdigest()
    class Environment:
        executed_action_frames=0
        def step(self,action):
            assert action==(0 if self.executed_action_frames==0 else 7)
            self.executed_action_frames+=4
            return frame,1.,self.executed_action_frames==8,False,{}
    monitor=SimpleNamespace(task=SimpleNamespace(result=lambda *_:dict(episode_score=2.)))
    rows=[dict(event='transition',run_step=tick,actions=[0 if tick==1 else 7],
        action_overrides=[0 if tick==1 else None],guidance_frame_sha256=[digest if tick==1 else None],
        rewards=[1.],terminated=[tick==2],truncated=[False],executed_action_frames=[tick*4]) for tick in (1,2)]
    rows.append(dict(event='episode',stream=0,episode_return=2.,terminated=True,truncated=False))
    episodes,_=replay_qbert_guidance.replay_rows([json.dumps(r) for r in rows],
        [Environment()],[monitor],[frame],QbertGuidance(bounded_config(1),1,18),float('inf'))
    assert episodes[0]['guided'] is True
