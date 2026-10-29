import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))
from probe_breakout_representation import (
    PROTOCOL, TARGETS, advance, fit_readouts, read_features, row_split, sha, validate_corpus,
)


def corpus_fixture():
    streams = []
    counts = dict(train=0, validation=0, test=0)
    for lane in range(8):
        records = []
        for index in range(18):
            episode, step = divmod(index, 9)
            split = row_split(lane, episode, step)
            if split:
                counts[split] += 1
            records.append(dict(action=index % 18, reward=float(index), episode=episode, step=step,
                split=split, phase='near_descending' if step == 8 else 'ball_unseen',
                frames=(index+1)*4, terminated=step == 8, truncated=False, reset_after=step == 8,
                input_pixel_sha256=hashlib.sha256(f'{lane}/{index}'.encode()).hexdigest(),
                targets=[step+lane, step*2, lane*2, None if step == 0 else 1,
                         None if step == 0 else -1, step+lane+episode]))
        streams.append(records)
    return dict(status='complete', targets=TARGETS, streams=streams, counts=counts,
        source_header=dict(num_envs=8, environment='ALE/Breakout-v5', mode='evaluate_greedy', config=dict(action_count=18)),
        source_end=dict(run_step=144, executed_action_frames=[72]*8, emulator_resets=[3]*8))


def capture_fixture(tmp_path, corpus):
    directory = tmp_path/'capture'
    directory.mkdir()
    files = []
    for lane, rows in enumerate(corpus['streams']):
        indices = np.array([i for i, row in enumerate(rows) if row['split']], np.int32)
        features = np.array([[rows[i]['step'], lane, rows[i]['episode'], 1.] for i in indices], np.float32)
        path = directory/f'stream-{lane}.npz'
        np.savez(path, indices=indices, cnn=features, posterior=features*2)
        digest = hashlib.sha256()
        for row in rows:
            digest.update(bytes.fromhex(row['input_pixel_sha256']))
        files.append(dict(stream=lane, file=path.name, sha256=sha(path), input_hash_sequence_sha256=digest.hexdigest()))
    result = dict(status='complete', protocol=PROTOCOL, smoke=False, actor_updates=0,
                  diagnostic_actions=144, files=files)
    (directory/'result.json').write_text(json.dumps(result))
    return directory


def test_lane_split_and_stride_ignore_target_presence():
    corpus = corpus_fixture()
    assert validate_corpus(corpus) == dict(train=16, validation=8, test=8)
    for rows in corpus['streams']:
        for row in rows:
            row['targets'] = [None]*6
    assert validate_corpus(corpus) == dict(train=16, validation=8, test=8)
    assert row_split(7, 3, 0) is None


@pytest.mark.parametrize('field,value', [('split', 'test'), ('step', 7), ('frames', 0),
    ('targets', [0, 0, 0, 1, 1, 0]), ('targets', [float('nan')]*6),
    ('reset_after', True), ('truncated', True)])
def test_invalid_alignment_or_reset_labels_fail(field, value):
    corpus = corpus_fixture()
    corpus['streams'][0][0][field] = value
    with pytest.raises(ValueError):
        validate_corpus(corpus)


def test_missing_terminal_reset_fails():
    corpus = corpus_fixture()
    corpus['streams'][0][8]['reset_after'] = False
    with pytest.raises(ValueError, match='lacks recorded reset'):
        validate_corpus(corpus)


def test_readout_alignment_and_missing_targets(tmp_path):
    corpus = corpus_fixture()
    directory = capture_fixture(tmp_path, corpus)
    data = read_features(corpus, directory)
    assert np.isnan(data['targets'][:, 3]).sum() == 16
    assert data['cnn'].shape == data['posterior'].shape == (32, 4)
    np.testing.assert_array_equal(data['cnn'][:, 1], data['lane'])
    assert data['index'].tolist() == [0, 8, 9, 17]*8


def test_reordered_features_fail_even_with_matching_file_checksum(tmp_path):
    corpus = corpus_fixture()
    directory = capture_fixture(tmp_path, corpus)
    path = directory/'stream-0.npz'
    with np.load(path) as values:
        data = {key: value.copy() for key, value in values.items()}
    data['indices'] = data['indices'][::-1]
    np.savez(path, **data)
    result = json.loads((directory/'result.json').read_text())
    result['files'][0]['sha256'] = sha(path)
    (directory/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='row alignment'):
        read_features(corpus, directory)


def test_readout_selection_and_baseline_ignore_test_targets(tmp_path):
    corpus = corpus_fixture()
    data = read_features(corpus, capture_fixture(tmp_path, corpus))
    first, predictions = fit_readouts(data)
    data['targets'][data['split'] == 'test'] += 1000
    second, changed_predictions = fit_readouts(data)
    assert first['training_mean'] == second['training_mean']
    for name in ('cnn', 'posterior'):
        assert first['variants'][name]['selection'] == second['variants'][name]['selection']
        np.testing.assert_array_equal(predictions[name], changed_predictions[name])
    assert first['variants']['cnn']['metrics']['all']['ball_x']['mae'] < .1
    assert first['variants']['cnn']['metrics']['near_descending']['dx']['count'] == 4


@pytest.mark.parametrize('last,terminal', [(False, False), (True, False), (False, True), (True, True)])
def test_recorded_and_collection_boundaries(last, terminal):
    class Actor:
        observed = []
        beginnings = []

        def act(self, *, action_mask):
            assert sum(action_mask) == 1
            return action_mask.index(True)

        def observe(self, frame, **kw):
            self.observed.append((frame, kw))

        def begin_episode(self, frame):
            self.beginnings.append(frame)

    class Environment:
        executed_action_frames = 4
        resets = 0

        def step(self, action):
            assert action == 3
            return 'arrival', 2., terminal, False, {}

        def reset(self):
            self.resets += 1
            return 'reset', {}

    row = dict(action=3, reward=2., terminated=terminal, truncated=False, frames=4, reset_after=terminal)
    original = copy.deepcopy(row)
    actor, environment = Actor(), Environment()
    frame, cut = advance(actor, environment, row, last=last, action_count=18)
    assert row == original
    assert cut == (last and not terminal)
    assert actor.observed == [('arrival', dict(extrinsic_reward=2., terminated=terminal, truncated=cut))]
    assert environment.resets == int(terminal)
    assert actor.beginnings == (['reset'] if terminal and not last else [])
    assert frame == ('reset' if terminal else 'arrival')


def test_fit_cli_never_imports_native_extension(tmp_path):
    corpus = corpus_fixture()
    corpus_path = tmp_path/'corpus.json'
    corpus_path.write_text(json.dumps(corpus))
    directory = capture_fixture(tmp_path, corpus)
    result_path = directory/'result.json'
    result = json.loads(result_path.read_text())
    result['corpus_sha256'] = sha(corpus_path)
    result_path.write_text(json.dumps(result))
    code = """
import importlib.abc,sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'kindle._native':
            raise RuntimeError('native import forbidden during CPU fit')
sys.meta_path.insert(0,NoNative())
from pathlib import Path
from probe_breakout_representation import fit
fit(*map(Path,sys.argv[1:]))
assert 'kindle._native' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-c', code, str(corpus_path), str(directory), str(directory), str(tmp_path/'fit')],
        env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(Path(__file__).parents[1]/'examples'), *sys.path])},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path/'fit/result.json').read_text())['status'] == 'complete'
