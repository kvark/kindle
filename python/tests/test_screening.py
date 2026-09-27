import numpy as np
import pytest

from kindle._screening import mean_ci, pack_minatar, summarize_curves


@pytest.mark.parametrize("channels", [1, 4, 7, 10, 16])
def test_minatar_packing_preserves_every_location_and_channel(channels):
    state = np.arange(100 * channels, dtype=np.float32).reshape(10, 10, channels)
    packed = pack_minatar(state).reshape(7, 7, 64)
    unpacked = packed[:5, :5, :4 * channels].reshape(5, 5, 2, 2, channels).transpose(0, 2, 1, 3, 4).reshape(state.shape)
    np.testing.assert_array_equal(state, unpacked)
    assert not packed[5:].any() and not packed[:, 5:].any() and not packed[:, :, 4 * channels:].any()


def test_minatar_packing_rejects_bad_shapes_and_nonfinite_values():
    for shape in [(10, 10), (10, 11, 3), (10, 10, 17), (10, 10, 0)]:
        with pytest.raises(ValueError):
            pack_minatar(np.zeros(shape))
    with pytest.raises(ValueError):
        pack_minatar(np.full((10, 10, 3), np.nan))


def test_bootstrap_is_over_seeds_and_time_never_extrapolates():
    result = mean_ci([1., 2., 3.])
    assert result["mean"] == 2.0 and result["ci95"] == [1.0, 3.0]
    runs = [dict(seed=i, curve=[dict(actions=step * 100, seconds=step * (i + 1), score=float(step + i))
                               for step in [1, 2, 3, 4]]) for i in range(3)]
    summary = summarize_curves(runs)
    assert [p["score"]["mean"] for p in summary["by_actions"]] == [2., 3., 4., 5.]
    assert summary["by_time"][0]["seconds"] == 3.0
    assert summary["by_time"][-1]["seconds"] == 4.0
    with pytest.raises(ValueError):
        summarize_curves(runs[:2])
    with pytest.raises(ValueError):
        summarize_curves([runs[0]] * 3)
