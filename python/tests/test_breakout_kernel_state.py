import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples'))
from probe_breakout_kernel_actions import kernel_probe
from probe_breakout_kernel_state import fit_state
from probe_breakout_representation import read_features
from test_breakout_representation import capture_fixture, corpus_fixture


def test_masked_targets_match_separate_observed_label_fit(tmp_path):
    corpus = corpus_fixture()
    data = read_features(corpus, capture_fixture(tmp_path, corpus))
    result, predictions = fit_state(data)
    tr = (data['split'] == 'train') & np.isfinite(data['targets'][:, 3])
    va = (data['split'] == 'validation') & np.isfinite(data['targets'][:, 3])
    te = data['split'] == 'test'
    expected, selection = kernel_probe(data['cnn'][tr], data['targets'][tr, 3:4],
        data['cnn'][va], data['targets'][va, 3:4], data['cnn'][te])
    np.testing.assert_array_equal(predictions['cnn'][:, 3], expected[:, 0])
    observed = result['variants']['cnn']['selection'][3]
    assert observed['training_count'] == 8 and observed['validation_count'] == 4
    assert observed['selection'] == selection['selection']
    assert result['variants']['cnn']['metrics']['all']['dx']['count'] == 4
    assert predictions['cnn'].shape == (8, 6)


def test_test_labels_and_missingness_do_not_change_fits(tmp_path):
    corpus = corpus_fixture()
    data = read_features(corpus, capture_fixture(tmp_path, corpus))
    first, predictions = fit_state(data)
    data['targets'][data['split'] == 'test'] = np.arange(6) + 1000
    second, changed = fit_state(data)
    for name in predictions:
        assert first['variants'][name]['selection'] == second['variants'][name]['selection']
        np.testing.assert_array_equal(predictions[name], changed[name])


def test_lane_overlap_and_insufficient_labels_fail(tmp_path):
    corpus = corpus_fixture()
    data = read_features(corpus, capture_fixture(tmp_path, corpus))
    data['targets'][data['split'] == 'validation', 0] = np.nan
    with pytest.raises(ValueError, match='insufficient'):
        fit_state(data)
    data['lane'][data['split'] == 'test'] = 0
    with pytest.raises(ValueError, match='lane roles'):
        fit_state(data)


def test_full_cpu_path_forbids_native_and_environment_imports(tmp_path):
    import json
    from probe_breakout_representation import sha
    corpus = corpus_fixture()
    path = tmp_path / 'corpus.json'
    path.write_text(json.dumps(corpus))
    directory = capture_fixture(tmp_path, corpus)
    result = json.loads((directory / 'result.json').read_text())
    result['corpus_sha256'] = sha(path)
    (directory / 'result.json').write_text(json.dumps(result))
    (tmp_path / 'final').symlink_to(directory, target_is_directory=True)
    (tmp_path / 'initial').symlink_to(directory, target_is_directory=True)
    code = """
import importlib.abc,sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in ('kindle._native','ale_py','gymnasium'):
            raise RuntimeError('native/environment import forbidden')
sys.meta_path.insert(0,NoNative())
from pathlib import Path
from probe_breakout_kernel_state import run
run(*map(Path,sys.argv[1:]))
assert 'kindle._native' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path), str(tmp_path/'fit')],
        env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(Path(__file__).parents[1]/'examples'), *sys.path])},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path/'fit/result.json').read_text())['status'] == 'complete'
