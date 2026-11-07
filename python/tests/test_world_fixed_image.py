"""Image-objective phase contracts; no GPU or emulator is constructed."""
import json
import sys
from pathlib import Path
import kindle
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))
import breakout_world_corpus as corpus


def metadata():
    return dict(format='kindle-world-only-v7',config=kindle.default_config(18,'tiny'),
        pixel_loss='motion_fixed_teacher_image',prediction_length=16,updates=12288,
        training_scope='fixed_teacher',phase_start_updates=8192,teacher_sha256='0'*64)


@pytest.mark.parametrize('version',[f'kindle-world-only-v{i}' for i in range(1,7)])
def test_image_phase_rejects_older_headers_before_gpu(tmp_path,version):
    m=metadata();m['format']=version;(tmp_path/'metadata.json').write_text(json.dumps(m))
    message={'kindle-world-only-v4':'invalid latent-only RGB checkpoint architecture',
             'kindle-world-only-v5':'invalid previous-frame RGB checkpoint architecture'}.get(version,
                 'requires world-only checkpoint format v7')
    with pytest.raises(RuntimeError,match=message):kindle.WorldModelLearner.restore(str(tmp_path))


def test_image_phase_requires_explicit_source():
    with pytest.raises(ValueError,match='explicit source phase'):
        kindle.WorldModelLearner(kindle.default_config(18,'tiny'),prediction_length=16,pixel_loss='motion_fixed_teacher_image')


def test_image_phase_checks_anchor_hash_before_loading_world_or_gpu(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps(metadata()));(tmp_path/'teacher.safetensors').write_bytes(b'wrong anchor')
    with pytest.raises(RuntimeError,match='anchor SHA-256 mismatch'):kindle.WorldModelLearner.restore(str(tmp_path))


def test_image_phase_report_total_uses_rgb_suffix_and_no_latent_or_energy_report():
    names=corpus.report_names(corpus.FIXED_IMAGE_OBJECTIVE,16);assert len(names)==11
    assert names[9:] == ('fixed_teacher_rgb_rollout','fixed_teacher_valid_fraction')
    values=[0,2,3,4,5,6,0,7,0,8,1]
    config=dict(loss_scales=dict(reconstruction=100,dynamics=1,representation=.1,reward=1,continuation=1))
    assert corpus.reported_total(config,values,corpus.FIXED_IMAGE_OBJECTIVE,16)==100*(2+8)+3+.1*4+5+6
    with pytest.raises(ValueError):corpus.report_names(corpus.FIXED_IMAGE_OBJECTIVE,0)
