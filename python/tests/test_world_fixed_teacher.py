"""Fixed-teacher contract checks run without constructing a GPU model."""
import json
import sys
from pathlib import Path
import kindle
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))
import breakout_world_corpus as corpus


def metadata():
    return dict(format='kindle-world-only-v6',config=kindle.default_config(18,'tiny'),
        pixel_loss='motion_fixed_teacher',prediction_length=16,updates=12288,
        training_scope='fixed_teacher',phase_start_updates=8192,teacher_sha256='0'*64)


@pytest.mark.parametrize('version',[f'kindle-world-only-v{i}' for i in range(1,6)])
def test_fixed_teacher_cannot_hide_in_older_format(tmp_path,version):
    m=metadata();m['format']=version;(tmp_path/'metadata.json').write_text(json.dumps(m))
    message = {'kindle-world-only-v4':'invalid latent-only RGB checkpoint architecture',
               'kindle-world-only-v5':'invalid previous-frame RGB checkpoint architecture'}.get(version,
                   'requires world-only checkpoint format v6')
    with pytest.raises(RuntimeError,match=message):
        kindle.WorldModelLearner.restore(str(tmp_path))


def test_fixed_teacher_requires_phase_constructor():
    with pytest.raises(ValueError,match='explicit source phase'):
        kindle.WorldModelLearner(kindle.default_config(18,'tiny'),prediction_length=16,pixel_loss='motion_fixed_teacher')


def test_teacher_hash_rejection_precedes_world_file_and_gpu(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps(metadata()))
    (tmp_path/'teacher.safetensors').write_bytes(b'wrong anchor')
    with pytest.raises(RuntimeError,match='anchor SHA-256 mismatch'):
        kindle.WorldModelLearner.restore(str(tmp_path))


def test_fixed_teacher_reports_use_both_existing_rollout_objectives():
    names=corpus.report_names(corpus.FIXED_TEACHER_OBJECTIVE,16);assert len(names)==14
    values=[0,2,3,4,5,6,0,7,0,8,9,10,2,1]
    config=dict(loss_scales=dict(reconstruction=100,dynamics=1,representation=.1,reward=1,continuation=1))
    assert corpus.reported_total(config,values,corpus.FIXED_TEACHER_OBJECTIVE,16)==100*(2+9)+3+.1*4+5+6+8
    with pytest.raises(ValueError):corpus.report_names(corpus.FIXED_TEACHER_OBJECTIVE,0)
