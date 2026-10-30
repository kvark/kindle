import copy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))
from probe_breakout_actions import ACTIONS, attach_actions, classification_metrics, fit_actions


def data_fixture():
    lanes=np.repeat(np.arange(8),16)
    labels=np.tile(np.repeat(ACTIONS,4),8)
    features=(labels[:,None]==ACTIONS).astype(float)
    return dict(lane=lanes,index=np.tile(np.arange(16),8),
        split=np.where(lanes<4,'train',np.where(lanes<6,'validation','test')),
        phase=np.where(labels==1,'ball_unseen','near_descending'),
        cnn=features,posterior=features*2),labels


def test_exact_confusion_does_not_collapse_action_aliases():
    result=classification_metrics(np.array([0,1,3,3,4]),np.array([0,1,6,3,4]))
    assert result['correct']==4 and result['agreement']==.8
    assert result['per_class']['3']==dict(count=2,correct=1,recall=.5)
    assert result['confusion'][2][6]==1 and result['macro_recall']==.875


def test_absent_classes_and_empty_slice_remain_explicit():
    result=classification_metrics(np.array([0,0]),np.array([0,3]))
    assert result['classes_with_support']==1 and result['macro_recall']==.5
    assert result['per_class']['1']==dict(count=0,correct=0,recall=None)
    empty=classification_metrics(np.array([],int),np.array([],int))
    assert empty['agreement'] is None and empty['macro_recall'] is None and empty['count']==0


@pytest.mark.parametrize('teacher,prediction',[([2],[0]),([0],[18]),([0],[.5]),([0,1],[0])])
def test_invalid_actions_fail(teacher,prediction):
    with pytest.raises(ValueError):
        classification_metrics(np.array(teacher),np.array(prediction))


def test_readout_selection_and_majority_do_not_use_test_labels():
    data,teacher=data_fixture()
    actual=np.full(len(teacher),0)
    first,predictions=fit_actions(data,teacher,actual)
    teacher[data['split']=='test']=4
    second,changed=fit_actions(data,teacher,actual)
    assert first['majority_action']==second['majority_action']==0
    assert first['training_class_counts']==second['training_class_counts']
    for name in ('cnn','posterior','combined'):
        assert first['variants'][name]['selection']==second['variants'][name]['selection']
        assert first['variants'][name]['metrics']['all']['agreement']==1
        np.testing.assert_array_equal(predictions[name],changed[name])
    assert first['variants']['combined']['width']==8


def test_phase_rows_align_by_identity_and_verify_unselected_inputs():
    corpus=dict(streams=[[dict(episode=0,step=0,action=3,phase='near_descending',input_pixel_sha256='x'),
                         dict(episode=0,step=1,action=4,phase='ascending',input_pixel_sha256='y')]])
    rows=[dict(stream=0,label=0,**corpus['streams'][0][0]),dict(stream=0,label=4,**corpus['streams'][0][1])]
    data=dict(lane=np.array([0]),index=np.array([0]))
    teacher,actual=attach_actions(corpus,data,rows[::-1])
    assert teacher.tolist()==[0] and actual.tolist()==[3]
    changed=copy.deepcopy(rows);changed[1]['input_pixel_sha256']='wrong'
    with pytest.raises(ValueError,match='alignment'):
        attach_actions(corpus,data,changed)
    with pytest.raises(ValueError,match='duplicate'):
        attach_actions(corpus,data,[rows[0],rows[0]])


def test_cpu_fit_forbids_native_import():
    code="""
import importlib.abc,sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='kindle._native':raise RuntimeError('native import forbidden')
sys.meta_path.insert(0,NoNative())
from test_breakout_action_readout import data_fixture
from probe_breakout_actions import fit_actions
data,teacher=data_fixture()
fit_actions(data,teacher,teacher)
assert 'kindle._native' not in sys.modules
"""
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30,
        env={**os.environ,'PYTHONPATH':os.pathsep.join([str(Path(__file__).parent),str(Path(__file__).parents[1]/'examples'),*sys.path])})
    assert result.returncode==0,result.stderr
