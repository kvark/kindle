from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))
import breakout_world_corpus as corpus


@pytest.mark.parametrize("objective",corpus.PREDICTIVE_OBJECTIVES)
def test_predictive_reports_keep_observed_indices_and_add_energy_diagnostics(objective):
    assert corpus.report_names('motion_cross_entropy',0)==corpus.LOSS_NAMES[:9]
    assert corpus.report_names('cross_entropy',16)==corpus.LOSS_NAMES
    names=corpus.report_names(objective,0)
    assert names[:9]==corpus.LOSS_NAMES[:9] and names[9:]==(
        'predictive_joint_energy','predictive_target_distance','predictive_sample_distance','predictive_valid_fraction')
    scales=dict(reconstruction=100.,dynamics=1.,representation=.1,reward=2.,continuation=3.)
    values=[0.,1.,2.,3.,4.,5.,0.,0.,0.,6.,7.,8.,.5]
    actual=corpus.reported_total(dict(loss_scales=scales),values,objective,0)
    assert actual==100*(1+6)+2+.1*3+2*4+3*5
    with pytest.raises(ValueError):corpus.report_names(objective,16)


def test_latent_rollout_uses_dynamics_scale_and_distinct_report_names():
    names=corpus.report_names(corpus.LATENT_OBJECTIVE,16)
    assert names[:9]==corpus.LOSS_NAMES[:9]
    assert names[9:]==('latent_overshooting_kl','latent_rollout_valid_fraction')
    with pytest.raises(ValueError):corpus.report_names(corpus.LATENT_OBJECTIVE,0)
    scales=dict(reconstruction=100.,dynamics=2.,representation=.1,reward=3.,continuation=4.)
    values=[0.,1.,2.,3.,4.,5.,0.,0.,0.,6.,.5]
    assert corpus.reported_total(dict(loss_scales=scales),values,corpus.LATENT_OBJECTIVE,16)==100+2*(2+6)+.1*3+3*4+4*5


def test_recurrent_energy_reports_and_total_keep_suffix_and_pixel_scales_distinct():
    names=corpus.report_names(corpus.RECURRENT_OBJECTIVE,16)
    assert names[:9]==corpus.LOSS_NAMES[:9]
    assert names[9:]==('recurrent_energy','recurrent_target_distance','recurrent_sample_distance','recurrent_valid_fraction')
    with pytest.raises(ValueError):corpus.report_names(corpus.RECURRENT_OBJECTIVE,0)
    values=[0.,1.,2.,3.,4.,5.,0.,0.,0.,.25,.5,.5,.75]
    scales=dict(reconstruction=100.,dynamics=2.,representation=.1,reward=3.,continuation=4.)
    assert corpus.reported_total(dict(loss_scales=scales),values,corpus.RECURRENT_OBJECTIVE,16)==100*(1+.25)+2*2+.1*3+3*4+4*5


def test_motion_suffix_reports_preserve_pixel_scale_and_full_suffix_mask():
    assert corpus.report_names('motion_cross_entropy',16)==corpus.LOSS_NAMES
    values=[0.,1.,2.,3.,4.,5.,0.,0.,0.,6.,.5]
    scales=dict(reconstruction=100.,dynamics=2.,representation=.1,reward=3.,continuation=4.)
    assert corpus.reported_total(dict(loss_scales=scales),values,'motion_cross_entropy',16)==700+2*2+.1*3+3*4+4*5


class PersistenceLearner:
    config = dict(batch_size=2, horizon=333)
    updates = 7
    pixel_loss = 'squared_error'

    def predict(self, observations, actions, future, *, seed, include_features):
        batch = self.config['batch_size']
        assert len(future) == 32*batch and min(future) >= 0
        assert actions[:batch] == [-1]*batch
        history = np.frombuffer(observations, '<f4').reshape(16, batch, 3, 64, 64)
        gamma = float(np.float32(1)-np.float32(1)/np.float32(333))
        result = dict(posterior_rgb=history[-1].tobytes(),
            prior_rgb=np.broadcast_to(history[-1], (32, batch, 3, 64, 64)).tobytes(),
            reward=np.zeros((32, batch)).tolist(), reward_positive_probability=np.zeros((32, batch)).tolist(),
            continuation=np.full((32, batch), gamma).tolist())
        if include_features:
            result.update(encoder_features=np.zeros((batch, 4), '<f4').tobytes(),
                posterior_features=np.zeros((batch, 5), '<f4').tobytes(),
                prior_features=np.zeros((32, batch, 5), '<f4').tobytes())
        return result


def stream():
    data = dict(rgb=np.zeros((60, 3, 64, 64), np.uint8), action=np.full(60, 2),
        reward=np.zeros(60), terminal=np.zeros(60, bool), position=np.zeros((60, 3)), lives=np.ones(60))
    for index in range(60): data['rgb'][index, 0, 40, 20+index % 10] = 255
    data['action'][[0, 21]] = -1
    data['terminal'][20] = True
    data['reward'][20] = 7
    return data


def test_evaluation_counts_only_real_roots_keeps_endings_and_handles_empty_horizon(tmp_path):
    result = corpus.evaluate_group(PersistenceLearner(), [stream()], [(0, 0), (0, 4), (0, 22)], tmp_path, 'development-natural')
    one = result['horizons']['1']
    assert one['count'] == 3 and one['terminal']['positive'] == 1
    assert one['terminal']['balanced_brier'] == .5
    assert one['positive_reward']['positive'] == 1 and one['rewarded_mae'] == 7
    assert result['horizons']['4']['count'] == 2
    assert result['horizons']['8']['count'] == 1
    assert result['horizons']['32']['count'] == 0
    assert result['horizons']['32']['images'][0]['mse'] is None
    assert one['images'][0] == one['persistence']
    data = np.load(tmp_path/'development-natural.npz')
    assert data['prediction'].shape == (4, 5, 3, 3, 64, 64)
    assert data['prior_features'].shape == (5, 3, 5)


@pytest.mark.parametrize('supplied', [False, True])
def test_restore_comparison_axes_match_a_prefix_of_the_original_batch(tmp_path, monkeypatch, supplied):
    import kindle
    reference = tmp_path/'reference'
    output = reference/'evaluations/7'
    output.mkdir(parents=True)
    locations = [(0, 0), (0, 4), (0, 22)]
    corpus.evaluate_group(PersistenceLearner(), [stream()], locations, output, 'development-natural')
    (reference/'selections.json').write_text(json.dumps(dict(development=dict(natural=locations))))
    restored = tmp_path/'restore'
    restored.mkdir()
    loads = []
    def load(path):
        assert not supplied
        loads.append(path)
        return PersistenceLearner()
    monkeypatch.setitem(kindle.__dict__, 'WorldModelLearner', SimpleNamespace(restore=load))
    monkeypatch.setattr(corpus, 'require_device', lambda _: None)
    args = SimpleNamespace(reference=reference, checkpoint=tmp_path/'unused-checkpoint', output=restored)
    corpus.restore_check(args, PersistenceLearner.config, [stream()], learner=PersistenceLearner() if supplied else None)
    result = json.loads((restored/'result.json').read_text())
    assert result['exact_prediction_restore'] and result['source_origins'] == 2
    assert len(loads) == (0 if supplied else 1)


@pytest.mark.parametrize('inherited', [0, 8192])
@pytest.mark.parametrize('pixel_loss', ['motion_cross_entropy', corpus.FIXED_EVENTS_OBJECTIVE])
def test_training_counts_incremental_updates_and_uses_global_evaluation_paths(tmp_path, monkeypatch, inherited, pixel_loss):
    import kindle
    from safetensors.numpy import save_file
    config = dict(seed=41001, batch_size=2, batch_length=32, loss_scales=dict(
        reconstruction=100., dynamics=1., representation=.1, reward=1., continuation=1.))

    class Learner:
        def __init__(self): self.updates = inherited; self.config = config; self.pixel_loss = pixel_loss
        def save(self, path):
            path = Path(path); path.mkdir()
            save_file({'world.parameter': np.array([self.updates], np.float32)}, path/'world.safetensors')
        def learn(self, observed, action, reward, terminal):
            assert len(observed) == 32*2*3*64*64*4
            self.updates += 1
            keep = corpus.rollout_validity(np.asarray(action).reshape(32,2),16)
            values = [0.,1.,2.,3.,4.,5.,0.,2.,0.,6.,float(keep.mean())]
            if self.pixel_loss == corpus.FIXED_EVENTS_OBJECTIVE:
                values = [0.,1.,2.,3.,4.,5.,0.,2.,0.,6.,.25,.5,.5,float(keep.mean()),.2,.3]
            values[0] = corpus.reported_total(config, values, self.pixel_loss, 16)
            return values

    learner = Learner()
    source = tmp_path/'inherited'
    learner.save(str(source))
    output = tmp_path/'new'; output.mkdir()
    args = SimpleNamespace(output=output, reference=None, steps=2, prediction_length=16,
        pixel_loss=pixel_loss, eval_every=1, dense_fitting=1, dense_development=1)
    evaluated = []
    monkeypatch.setattr(corpus, 'require_device', lambda _: None)
    monkeypatch.setattr(corpus, 'evaluate_group', lambda model, data, locations, output, name, **kwargs:
        evaluated.append((model.updates, output, name)))
    monkeypatch.setitem(kindle.__dict__, 'WorldModelLearner', lambda *a, **kw: learner)
    corpus.train(args, config, [stream(),stream()], [stream(),stream()],
                 learner=learner if inherited else None, initial_checkpoint=source)
    result = json.loads((output/'result.json').read_text())
    assert result['updates'] == inherited+2 and result['start_updates'] == inherited and result['new_updates'] == 2
    assert {int(path.name) for _,path,_ in evaluated if path.parent.name == 'evaluations'} == {inherited,inherited+1,inherited+2}
    rows = [json.loads(line) for line in (output/'learning.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [1,2]
    assert [row['global_update'] for row in rows] == [inherited+1,inherited+2]


def test_zero_prior_horizon_keeps_sampling_and_counts_without_negative_zero_slices():
    streams = [stream(), stream()]
    config = dict(seed=41001, batch_size=2, batch_length=32)
    default_plan, default_counts = corpus.sampler_plan(streams, config, 32)
    zero_plan, zero_counts = corpus.sampler_plan(streams, config, 32, 0)
    assert np.array_equal(default_plan, zero_plan)
    assert default_counts['prior_terminal_rows'] > 0 and zero_counts['prior_terminal_rows'] == 0
    assert {k:v for k,v in default_counts.items() if k != 'prior_terminal_rows'} == {
        k:v for k,v in zero_counts.items() if k != 'prior_terminal_rows'}
    actions = np.array([[-1, -1], [0, 1], [1, -1], [2, 2]])
    assert corpus.rollout_validity(actions, 0).shape == (0, 2)
    assert np.array_equal(corpus.rollout_validity(actions, 2), [[True, False], [True, False]])
