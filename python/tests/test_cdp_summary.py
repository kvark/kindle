import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples'))
import summarize_cdp as summary


def header(method):
    cdp = method == 'cdp'
    return dict(environment='ALE/Seaquest-v5', num_envs=8, steps=200000, full_action_space=True,
                sticky_actions=.25, action_repeat=4, noop_max=0, max_episode_frames=100000,
                mode='train', observation_size='native', starting_environment_step=0,
                starting_learner_step=0, restored_checkpoint=None, seed=1009,
                environment_seeds=[1009 + i * 1000003 for i in range(8)],
                native_extension_sha256='native', runner_sha256='runner', wrapper_sha256='wrapper',
                learned_rgb_preprocessing='one GPU resize', model_provenance=dict(meganeura='same',
                    future_head_revision='cosine' if cdp else None),
                config=dict(model_size='size1_m', observation_kind='rgb64', video_encoder=None, action_count=18,
                    batch_size=8, batch_length=16, world_backprop_length=16, world_microbatch_size=8,
                    replay_context=1, replay_capacity=100000, train_ratio=32., imagination_length=15,
                    learning_rate=4e-5, learning_rate_warmup=1000, agc=.3, actor_unimix=0,
                    replay_value_gradient=True, actor_critic_gradient=False, intrinsic_reward_scale=0,
                    extrinsic_reward_scale=1, visitation_bonus=False, seed=1009,
                    encoder_learning_rate=6e-6 if cdp else None, dynamics_learning_rate=4e-4 if cdp else None,
                    loss_scales=dict(future_prediction=500 if cdp else 0, reconstruction=0 if cdp else 1,
                                     dynamics=1, representation=.1)))


def test_shared_recipe_excludes_only_declared_loss_rate_differences():
    assert summary.recipe(header('rgb'), 'rgb') == summary.recipe(header('cdp'), 'cdp')
    changed = header('cdp')
    changed['config']['loss_scales']['representation'] = .2
    assert summary.recipe(header('rgb'), 'rgb') != summary.recipe(changed, 'cdp')
    changed = header('rgb')
    changed['model_provenance']['meganeura'] = 'other'
    assert summary.recipe(header('cdp'), 'cdp') != summary.recipe(changed, 'rgb')


@pytest.mark.parametrize('change', ['label', 'budget', 'sticky', 'aid', 'rate', 'schedule'])
def test_recipe_refuses_mismatches(change):
    value = header('cdp')
    if change == 'label':
        method = 'rgb'
    else:
        method = 'cdp'
        if change == 'budget':
            value['steps'] = 200008
        elif change == 'sticky':
            value['sticky_actions'] = 0
        elif change == 'aid':
            value['exploration'] = {'override': True}
        elif change == 'rate':
            value['config']['encoder_learning_rate'] = None
        else:
            value['config']['train_ratio'] = 16
    with pytest.raises(ValueError):
        summary.recipe(value, method)


@pytest.mark.parametrize('failure', [None, 'config', 'counter', 'hash', 'nonfinite'])
def test_checkpoint_audit_checks_saved_recipe_counters_identity_and_finiteness(tmp_path, failure):
    h = header('cdp')
    metadata = dict(config=copy.deepcopy(h['config']), learner_step=49939, tensor_sha256={})
    for name in ('world', 'behavior', 'slow_value'):
        path = tmp_path / f'{name}.safetensors'
        value = np.array([np.nan if failure == 'nonfinite' else 1.], np.float32)
        save_file({'weight': value}, path)
        metadata['tensor_sha256'][name] = summary.sha256_file(path)
    if failure == 'config':
        metadata['config']['agc'] = .1
    elif failure == 'counter':
        metadata['learner_step'] = 49938
    elif failure == 'hash':
        metadata['tensor_sha256']['world'] = 'wrong'
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    if failure:
        with pytest.raises(ValueError):
            summary.checkpoint_audit(tmp_path, h, 49939)
    else:
        assert summary.checkpoint_audit(tmp_path, h, 49939)['finite_tensor_counts'] == dict(world=1, behavior=1, slow_value=1)


def test_partial_summary_never_reads_active_or_failed_jobs(tmp_path):
    assert summary.summarize(tmp_path)['completed_runs'] == 0
    path = tmp_path / 'learning-queue/seaquest-rgb-1009'
    path.mkdir(parents=True)
    (path / 'result.json').write_text(json.dumps(dict(host_guard_passed=False)))
    with pytest.raises(ValueError, match='failed guard'):
        summary.summarize(tmp_path)


@pytest.mark.parametrize('failure', [None, 'reset', 'origin', 'cut', 'alignment', 'nonfinite'])
def test_saved_trace_audit_reconstructs_boundaries_and_alignment(failure):
    from test_cdp_probes import trace
    data = trace()
    if failure == 'reset':
        data['episodes'][20] = 0
    elif failure == 'origin':
        data['origins_h15'][0] = 16
    elif failure == 'cut':
        data['collection_cut'][-1] = False
    elif failure == 'alignment':
        data['prior_h1'][0, 0] += 1
    elif failure == 'nonfinite':
        data['cnn'][0, 0] = np.nan
    if failure:
        with pytest.raises(ValueError):
            summary.audit_trace(data, 65, deter=2)
    else:
        assert summary.audit_trace(data, 65, deter=2) == 0
