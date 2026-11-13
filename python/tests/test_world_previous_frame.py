"""Reject incompatible previous-frame checkpoints before native construction."""
import json
import kindle
import pytest


def test_previous_frame_remains_off_in_default_config():
    assert 'rgb_previous_frame' not in kindle.default_config(18, 'tiny')


@pytest.mark.parametrize('version', ['kindle-world-only-v1','kindle-world-only-v2','kindle-world-only-v3','kindle-world-only-v4'])
def test_previous_frame_requires_its_architecture_version(tmp_path, version):
    c=kindle.default_config(18,'tiny');c['rgb_previous_frame']=True
    (tmp_path/'metadata.json').write_text(json.dumps(dict(format=version,config=c,updates=0,
        prediction_length=0,pixel_loss='motion_cross_entropy')))
    with pytest.raises(RuntimeError,match='requires world-only checkpoint format v5'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('flag',[False,None])
def test_v5_requires_previous_frame_architecture(tmp_path,flag):
    c=kindle.default_config(18,'tiny')
    if flag is not None:c['rgb_previous_frame']=flag
    (tmp_path/'metadata.json').write_text(json.dumps(dict(format='kindle-world-only-v5',config=c,
        updates=0,prediction_length=0,pixel_loss='motion_cross_entropy')))
    with pytest.raises(RuntimeError,match='invalid previous-frame RGB checkpoint architecture'):
        kindle.WorldModelLearner.restore(str(tmp_path))


def test_previous_frame_does_not_enter_legacy_frozen_visual_phase(tmp_path):
    c=kindle.default_config(18,'tiny');c['rgb_previous_frame']=True
    (tmp_path/'metadata.json').write_text(json.dumps(dict(format='kindle-world-only-v5',config=c,
        updates=8192,prediction_length=0,pixel_loss='motion_cross_entropy')))
    with pytest.raises(RuntimeError,match='requires single-frame RGB'):
        kindle.WorldModelLearner.start_frozen_visual_phase(str(tmp_path))
