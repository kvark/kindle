"""Fixed-filter future-event API contracts; no GPU or emulator construction."""
import json
import sys
from pathlib import Path
import kindle
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))
import breakout_world_corpus as corpus


def metadata():
    return dict(format='kindle-world-only-v16',config=kindle.default_config(18,'tiny'),
        pixel_loss='motion_fixed_teacher_events',prediction_length=16,updates=12288,
        training_scope='fixed_teacher',phase_start_updates=8192,teacher_sha256='0'*64)


@pytest.mark.parametrize('version', range(1,16))
def test_fixed_events_reject_older_formats_before_loading_world_or_gpu(tmp_path,version):
    m=metadata();m['format']=f'kindle-world-only-v{version}'
    (tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError):
        kindle.WorldModelLearner.restore(str(tmp_path))


def test_fixed_events_require_explicit_phase_constructor():
    with pytest.raises(ValueError,match='explicit source phase'):
        kindle.WorldModelLearner(kindle.default_config(18,'tiny'),prediction_length=16,
            pixel_loss=corpus.FIXED_EVENTS_OBJECTIVE)


def test_fixed_events_reject_anchor_digest_before_world_or_gpu(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps(metadata()))
    (tmp_path/'teacher.safetensors').write_bytes(b'wrong anchor')
    with pytest.raises(RuntimeError,match='anchor SHA-256 mismatch'):
        kindle.WorldModelLearner.restore(str(tmp_path))


def test_fixed_events_reports_add_both_future_heads_with_configured_scales():
    names=corpus.report_names(corpus.FIXED_EVENTS_OBJECTIVE,16)
    assert names[:14]==corpus.report_names(corpus.FIXED_TEACHER_OBJECTIVE,16)
    assert names[14:]==('fixed_teacher_future_reward','fixed_teacher_future_continuation')
    v=[0,2,3,4,5,6,0,7,0,8,9,10,2,1,11,12]
    c=dict(loss_scales=dict(reconstruction=100,dynamics=2,representation=.1,reward=3,continuation=4))
    assert corpus.reported_total(c,v,corpus.FIXED_EVENTS_OBJECTIVE,16)==100*(2+9)+2*(3+8)+.1*4+3*(5+11)+4*(6+12)
    with pytest.raises(ValueError):corpus.report_names(corpus.FIXED_EVENTS_OBJECTIVE,0)
