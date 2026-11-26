import copy
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples'))
from analyze_gradient_optimizer import TERMS, effective_rate, group, matrix_summary, transform


def config():
    return dict(learning_rate=.1, learning_rate_warmup=0, encoder_learning_rate=None,
                dynamics_learning_rate=None, optimizer_beta1=.5, optimizer_beta2=.75,
                optimizer_epsilon=1e-20, agc=0, agc_pmin=.001)


def gradients(total):
    values = np.zeros((len(TERMS), len(total)))
    values[0] = values[-1] = total
    return values


def test_first_laprop_step_is_bias_corrected_normalized_gradient():
    stages, info = transform([3, 4], [0, 0], [0, 0], gradients([2, -4]), config(), 'weight', 1, 0)
    np.testing.assert_allclose(stages['updates'][0], [-.1, .1], rtol=1e-7)
    np.testing.assert_array_equal(stages['updates'][1], [0, 0])
    assert info['agc_scale'] == 1


def test_saved_momentum_can_reverse_current_gradient():
    cfg = config()
    # v'=4, corrected variance=4/(1-.75**2); g/rms=sqrt(1-.75**2).
    stages, _ = transform([1], [-2], [4], gradients([2]), cfg, 'weight', 2, 1)
    expected = -.1 * (-1 + .5 * math.sqrt(1 - .75 ** 2)) / (1 - .5 ** 2)
    assert stages['updates'][0, 0] > 0 > stages['updates'][2, 0]
    np.testing.assert_allclose(stages['updates'][0, 0], expected, rtol=1e-7)


def test_canceling_losses_share_total_clipping_and_denominator():
    cfg = config(); cfg['agc'] = .1
    raw = gradients([1]); raw[1] = 100; raw[-1] = -99
    stages, info = transform([10], [0], [0], raw, cfg, 'weight', 1, 0)
    assert info['agc_scale'] == 1
    np.testing.assert_allclose(stages['updates'][3:].sum(axis=0), stages['updates'][2], atol=1e-13)
    assert abs(stages['updates'][3, 0]) > abs(stages['updates'][2, 0]) * 99


def test_clipping_is_per_tensor_and_uses_weight_norm_floor():
    cfg = config(); cfg.update(agc=.5, agc_pmin=2)
    _, small = transform([0, 0], [0, 0], [1, 1], gradients([3, 4]), cfg, 'a', 2, 1)
    _, large = transform([30, 40], [0, 0], [1, 1], gradients([3, 4]), cfg, 'b', 2, 1)
    assert small['agc_scale'] == .2 and large['agc_scale'] == 1


def test_zero_gradient_retains_history_without_division_by_zero():
    cfg = config()
    stages, info = transform([1, 0], [1, -1], [0, 1], gradients([0, 0]), cfg, 'a', 3, 2)
    assert info['agc_scale'] == 1
    np.testing.assert_array_equal(stages['updates'][0], stages['updates'][1])
    np.testing.assert_array_equal(stages['updates'][2:], 0)
    assert np.isfinite(stages['updates']).all()


def test_learning_rates_apply_warmup_and_exact_prefixes():
    cfg = config(); cfg.update(learning_rate=4e-5, learning_rate_warmup=1000,
                              encoder_learning_rate=6e-6, dynamics_learning_rate=4e-4)
    for name, expected in [('world.representation.encoder.conv.weight', 3e-6),
                           ('world.representation.posterior.weight', 2e-4),
                           ('world.dynamics.weight', 2e-4), ('world.future_predictor.weight', 2e-4),
                           ('world.exploration.weight', 2e-4), ('world.reward.weight', 2e-5),
                           ('world.representation.encoderish.weight', 2e-5)]:
        assert effective_rate(cfg, name, 500) == pytest.approx(expected, rel=1e-7)
    assert effective_rate(cfg, 'world.dynamics.weight', 0) == 0
    assert effective_rate(cfg, 'world.dynamics.weight', 2000) == pytest.approx(4e-4, rel=1e-7)


@pytest.mark.parametrize('field,value', [('variance', [-1]), ('momentum', [float('nan')]),
                                       ('parameter', [float('inf')]), ('step', 0)])
def test_invalid_saved_state_is_rejected(field, value):
    args = dict(parameter=[1], momentum=[0], variance=[0], gradients=gradients([1]),
                config=config(), name='a', step=1, learner_step=0)
    args[field] = value
    with pytest.raises(ValueError, match='invalid'):
        transform(**args)


def test_estimates_do_not_mutate_inputs_or_chain_batches():
    p, m, v, raw = np.array([1.]), np.array([.2]), np.array([.3]), gradients([2])
    original = [a.copy() for a in (p, m, v, raw)]
    cfg = config(); before = copy.deepcopy(cfg)
    first, _ = transform(p, m, v, raw, cfg, 'a', 3, 2)
    second, _ = transform(p, m, v, raw, cfg, 'a', 3, 2)
    for actual, expected in zip((p, m, v, raw), original, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(first['updates'], second['updates'])
    assert cfg == before


def test_zero_cosines_and_parameter_groups_remain_explicit():
    result = matrix_summary(np.array([[0., 0], [0, 4]]), ('zero', 'nonzero'))
    assert result['cosines'] == [[None, None], [None, 1]]
    assert [group(name) for name in ('world.representation.encoder.w',
            'world.representation.posterior.w', 'world.dynamics.w', 'world.reward.w')] == [
                'encoder', 'posterior', 'dynamics', 'other']
