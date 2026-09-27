from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from summarize_representation_probes import aggregate, columns


def test_report_separates_motion_and_preserves_negative_or_undefined_r2():
    fits = [dict(test=dict(all={"x": dict(r2=.8), "y": dict(r2=None), "x_velocity": dict(r2=-.2)})),
            dict(test=dict(all={"x": dict(r2=.6), "y": dict(r2=None), "x_velocity": dict(r2=-.4)}))]
    assert aggregate(fits) == pytest.approx(.7)
    assert aggregate(fits, motion=True) == pytest.approx(-.3)
    assert aggregate([dict(test=dict(all={"x": dict(r2=None)}))]) is None


def test_report_column_packing_preserves_target_order_and_counts():
    metrics = dict(second=dict(count=2, r2=None, mae=3., rmse=4.),
                   first=dict(count=10, r2=.5, mae=1., rmse=2.))
    assert columns(metrics, ["first", "second"]) == dict(count=[10, 2], r2=[.5, None], mae=[1., 3.], rmse=[2., 4.])
