import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))
from probe_breakout_actor import actor_probabilities, policy_metrics


def parameters():
    rng=np.random.default_rng(29)
    result={}
    for index in range(3):
        prefix=f'behavior.actor.layer{index}.'
        result[prefix+'weight']=rng.normal(size=(2 if index==0 else 3,3))
        result[prefix+'bias']=rng.normal(size=3)
        result[prefix+'norm.weight']=rng.normal(size=3)
    result['behavior.actor.out.weight']=rng.normal(size=(3,4))
    result['behavior.actor.out.bias']=rng.normal(size=4)
    return result


@pytest.mark.parametrize('unimix',[0.,.01,.1])
def test_reference_matches_independent_scalar_calculation(unimix):
    weights=parameters();inputs=[[0.,0.],[1.,-2.],[-3.,4.]]
    expected=[]
    for row in inputs:
        for index in range(3):
            prefix=f'behavior.actor.layer{index}.'
            projected=[sum(x*w for x,w in zip(row,column))+bias
                       for column,bias in zip(weights[prefix+'weight'].T,weights[prefix+'bias'])]
            rms=math.sqrt(sum(x*x for x in projected)/len(projected)+1e-4)
            row=[x/rms*scale for x,scale in zip(projected,weights[prefix+'norm.weight'])]
            row=[x/(1+math.exp(-x)) for x in row]
        logits=[sum(x*w for x,w in zip(row,column))+bias for column,bias in
                zip(weights['behavior.actor.out.weight'].T,weights['behavior.actor.out.bias'])]
        values=[math.exp(x-max(logits)) for x in logits]
        expected.append([(1-unimix)*x/sum(values)+unimix/len(values) for x in values])
    np.testing.assert_allclose(actor_probabilities(inputs,weights,unimix),expected,rtol=1e-13,atol=1e-14)


def test_probability_floor_and_extreme_output_bias_are_stable():
    weights=parameters();weights['behavior.actor.out.bias']=np.array([10000.,0.,-10000.,0.])
    result=actor_probabilities([[1.,-2.]],weights,.01)
    np.testing.assert_allclose(result,[[.9925,.0025,.0025,.0025]],atol=1e-15)


@pytest.mark.parametrize('kind',['missing','extra','nonfinite','hidden_shape','output_shape','features','unimix'])
def test_invalid_actor_input_fails(kind):
    weights=parameters();features=[[0.,1.]];unimix=.01
    if kind=='missing':weights.pop('behavior.actor.layer0.norm.weight')
    elif kind=='extra':weights['unexpected']=np.zeros(1)
    elif kind=='nonfinite':weights['behavior.actor.out.bias'][0]=np.nan
    elif kind=='hidden_shape':weights['behavior.actor.layer0.bias']=np.zeros(4)
    elif kind=='output_shape':weights['behavior.actor.out.weight']=np.zeros((4,4))
    elif kind=='features':features=[[np.nan,1.]]
    else:unimix=1.
    with pytest.raises(ValueError):actor_probabilities(features,weights,unimix)


def test_policy_metrics_count_direction_and_fire_errors_without_aliasing():
    teacher=np.array([3,4,3,1,0]);chosen=np.array([4,3,0,0,6])
    probabilities=np.full((5,18),.01/18)
    probabilities[np.arange(5),chosen]+=.99
    result=policy_metrics(probabilities,teacher,np.array([6,6,7,7,7]),np.array(['near_descending']*3+['ball_unseen']*2))
    near=result['near_descending'];all_rows=result['all']
    assert near['count']==3 and near['directional_teacher_count']==3
    assert near['opposite_direction_count']==2 and near['directional_noop_count']==1
    assert all_rows['fire_noop_count']==1 and all_rows['non_teacher_action_count']==1
    assert all_rows['correct']==0
    assert all_rows['mean_teacher_probability']==pytest.approx(.01/18)
    assert all_rows['teacher_cross_entropy']==pytest.approx(-math.log(.01/18))
