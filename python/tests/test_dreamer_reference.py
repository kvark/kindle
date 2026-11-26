import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from dreamer_step_reference import parameter_layout, upstream_parameter


@pytest.mark.parametrize("upstream,native", [
    ("enc/cnn0norm/scale", "world.representation.encoder.cnn0.norm.weight"),
    ("dec/spnorm/scale", "world.decoder.spatial.norm.weight"),
    ("dec/sp0/kernel", "world.decoder.sp0.weight"),
    ("dyn/dynin0norm/scale", "world.dynamics.core.dynin0.norm.weight"),
    ("dyn/obslogit/kernel", "world.representation.posterior.obslogit.weight"),
    ("dyn/prior0/bias", "world.dynamics.prior0.bias"),
    ("dyn/pred0/kernel", "world.future_predictor.layer0.weight"),
    ("dyn/pred0norm/scale", "world.future_predictor.layer0.norm.weight"),
    ("dyn/pred_out/bias", "world.future_predictor.out.bias"),
    ("rew/mlp/norm0/scale", "world.reward.layer0.norm.weight"),
    ("con/head/logit/kernel", "world.continuation.out.weight"),
    ("pol/head/action/logits/kernel", "behavior.actor.out.weight"),
    ("val/mlp/linear2/kernel", "behavior.value.layer2.weight"),
    ("slowval/head/logits/bias", "behavior.value.out.bias"),
])
def test_parameter_names_and_dense_storage(upstream, native):
    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    name, converted = parameter_layout(upstream, value)
    assert name == native
    np.testing.assert_array_equal(converted, value.reshape(-1))
    np.testing.assert_array_equal(upstream_parameter(upstream, converted, value.shape), value)


@pytest.mark.parametrize("name", ["enc/cnn0/kernel", "dec/conv2/kernel", "dec/imgout/kernel"])
def test_convolution_storage_roundtrip(name):
    value = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
    _, converted = parameter_layout(name, value)
    np.testing.assert_array_equal(converted, value.transpose(3, 2, 0, 1).reshape(-1))
    np.testing.assert_array_equal(upstream_parameter(name, converted, value.shape), value)
