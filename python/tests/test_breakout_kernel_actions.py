import copy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples'))
from probe_breakout_kernel_actions import fit_actions, kernel_probe
from test_breakout_action_readout import data_fixture


def test_centered_solver_matches_augmented_direct_system():
    rng = np.random.default_rng(137)
    x, v, t = (rng.normal(size=(n, 3)) for n in (12, 6, 5))
    y, vy = rng.normal(size=(12, 2)), rng.normal(size=(6, 2))
    prediction, selection = kernel_probe(x, y, v, vy, t, alphas=(.1,))
    scale = x.std(0) * np.sqrt(3)
    normalized = x / scale
    distances = np.square(normalized[:, None, :] - normalized[None, :, :]).sum(2)
    bandwidth = np.median(distances[np.triu_indices(len(x), k=1)])
    kernel = np.exp(-distances / bandwidth)
    system = np.block([[kernel + .1 * np.eye(len(x)), np.ones((len(x), 1))],
                       [np.ones((1, len(x))), np.zeros((1, 1))]])
    coefficients = np.linalg.solve(system, np.vstack([y, np.zeros((1, 2))]))
    cross = np.exp(-np.square((t[:, None, :] - x[None, :, :]) / scale).sum(2) / bandwidth)
    expected = cross @ coefficients[:-1] + coefficients[-1]
    np.testing.assert_allclose(prediction, expected, atol=1e-10)
    assert selection['bandwidth_squared'] == pytest.approx(bandwidth)


def test_xor_is_accessible_and_test_features_do_not_change_selection():
    x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], float)
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], float)
    first, selection = kernel_probe(x, y, x * .9, y, x * 1.1)
    assert np.array_equal(first.argmax(1), y.argmax(1))
    second, changed = kernel_probe(x, y, x * .9, y, np.vstack([x * 1.1, [1000, -1000]]))
    assert selection == changed
    np.testing.assert_allclose(first, second[:4])


def test_test_labels_do_not_affect_selection_or_predictions():
    data, teacher = data_fixture()
    first, predictions = fit_actions(data, teacher, teacher)
    changed_labels = teacher.copy()
    changed_labels[data['split'] == 'test'] = 4
    second, changed = fit_actions(data, changed_labels, teacher)
    assert first['training_class_counts'] == second['training_class_counts']
    for name in ('cnn', 'posterior', 'combined'):
        for key in ('selection', 'bandwidth_squared'):
            assert first['variants'][name][key] == second['variants'][name][key]
        assert first['variants'][name]['metrics']['all']['agreement'] == 1
        np.testing.assert_array_equal(predictions[name], changed[name])
    contaminated = copy.deepcopy(data)
    contaminated['lane'][data['split'] == 'test'] = 0
    with pytest.raises(ValueError, match='lane roles'):
        fit_actions(contaminated, teacher, teacher)


def test_constant_training_features_fail_explicitly():
    x, y = np.ones((4, 2)), np.eye(4)
    with pytest.raises(ValueError, match='constant training'):
        kernel_probe(x, y, x, y, x)


def test_cpu_readout_forbids_native_import():
    code = """
import importlib.abc, sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'kindle._native': raise RuntimeError('native import forbidden')
sys.meta_path.insert(0, NoNative())
from test_breakout_action_readout import data_fixture
from probe_breakout_kernel_actions import fit_actions
data, teacher = data_fixture()
fit_actions(data, teacher, teacher)
assert 'kindle._native' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30,
        env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(Path(__file__).parent),
             str(Path(__file__).parents[1] / 'examples'), *sys.path])})
    assert result.returncode == 0, result.stderr
