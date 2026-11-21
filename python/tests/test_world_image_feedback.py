"""Image-feedback metadata/objective checks; no GPU or emulator construction."""
import json
from pathlib import Path
import sys
import kindle
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))


def metadata():
    c=kindle.default_config(18,'tiny');c.update(rgb_spatial_decoder=True,world_image_feedback=True)
    return dict(format='kindle-world-only-v11',config=c,pixel_loss='motion_image_feedback',prediction_length=16,updates=0)


def test_image_feedback_is_absent_by_default():
    assert 'world_image_feedback' not in kindle.default_config(18,'tiny')


@pytest.mark.parametrize('version',range(1,11))
def test_image_feedback_requires_v11_before_gpu(tmp_path,version):
    m=metadata();m['format']=f'kindle-world-only-v{version}';(tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='image feedback requires checkpoint format v11'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('field',['world_image_feedback','rgb_spatial_decoder'])
def test_v11_requires_both_architecture_flags(tmp_path,field):
    m=metadata();m['config'][field]=False;(tmp_path/'metadata.json').write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match='invalid image feedback checkpoint architecture'):
        kindle.WorldModelLearner.restore(str(tmp_path))


@pytest.mark.parametrize('objective',['motion_cross_entropy','squared_error'])
def test_feedback_architecture_rejects_other_objectives(objective):
    with pytest.raises(ValueError,match='image feedback requires its explicit objective'):
        kindle.WorldModelLearner(metadata()['config'],prediction_length=16,pixel_loss=objective)


def test_feedback_requires_a_positive_suffix():
    with pytest.raises(ValueError,match='positive prediction length'):
        kindle.WorldModelLearner(metadata()['config'],pixel_loss='motion_image_feedback')


def test_feedback_report_counts_and_total_include_both_future_heads():
    import breakout_world_corpus as corpus
    names=corpus.report_names('motion_image_feedback',16)
    assert len(names)==13 and names[-2:]==('feedback_future_reward','feedback_future_continuation')
    values=list(range(13))
    c=dict(loss_scales=dict(reconstruction=100,dynamics=1,representation=0.1,reward=2,continuation=3))
    assert corpus.reported_total(c,values,'motion_image_feedback',16)==100*(1+9)+2+0.1*3+2*(4+11)+3*(5+12)
