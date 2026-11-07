import copy
import json
import sys
from types import SimpleNamespace

import pytest
import atari_vector
import kindle


@pytest.mark.parametrize('args,message', [
    (['--actor-hidden-units','0'],'positive unsigned'),
    (['--actor-hidden-units','-1'],'positive unsigned'),
    (['--actor-hidden-units',str(2**64)],'positive unsigned'),
    (['--actor-hidden-units','512','--restore','unused'],'requires fresh training'),
    (['--actor-hidden-units','512','--evaluate'],'requires fresh training'),
])
def test_invalid_width_stops_before_construction(monkeypatch,tmp_path,capsys,args,message):
    monkeypatch.setattr(atari_vector.gym,'make',lambda *_,**__:pytest.fail('environment must not start'))
    output=tmp_path/'never.jsonl'
    monkeypatch.setattr(sys,'argv',['atari_vector.py','--output',str(output),*args])
    with pytest.raises(SystemExit) as error: atari_vector.main()
    assert error.value.code==2 and not output.exists()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('restore',[False,True])
def test_width_rejects_old_native(monkeypatch,tmp_path,capsys,restore):
    monkeypatch.setattr(kindle,'VectorAgent',SimpleNamespace())
    monkeypatch.setattr(atari_vector.gym,'make',lambda *_,**__:pytest.fail('environment must not start'))
    args=['--actor-hidden-units','512']
    if restore:
        checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
        (checkpoint/'metadata.json').write_text(json.dumps(dict(config=dict(actor_hidden_units=512))))
        args=['--evaluate','--restore',str(checkpoint),'--observation-size','native']
    monkeypatch.setattr(sys,'argv',['atari_vector.py','--output',str(tmp_path/'never.jsonl'),*args])
    with pytest.raises(SystemExit) as error:atari_vector.main()
    assert error.value.code==2
    assert 'requires native actor width support' in capsys.readouterr().err


def test_width_only_changes_declared_config(monkeypatch,tmp_path):
    environment=SimpleNamespace(action_space=SimpleNamespace(n=18),reset=lambda **_:(None,{}),close=lambda:None)
    monkeypatch.setattr(atari_vector.gym,'make',lambda *_,**__:environment)
    monkeypatch.setattr(atari_vector,'DreamerAtariPreprocessing',lambda env,**_:env)
    configs=[]
    def construct(streams,config):
        configs.append(copy.deepcopy(config));raise RuntimeError('config inspected before GPU')
    monkeypatch.setattr(kindle,'VectorAgent',SimpleNamespace(learned_rgb=construct,SUPPORTS_ACTOR_HIDDEN_UNITS=True))
    for enabled in [False,True]:
        monkeypatch.setattr(sys,'argv',['atari_vector.py','--output',str(tmp_path/f'{enabled}.jsonl'),
            '--cdp','--cdp-centered',*(['--actor-hidden-units','512'] if enabled else [])])
        with pytest.raises(RuntimeError,match='config inspected before GPU'):atari_vector.main()
    plain,wide=configs
    assert 'actor_hidden_units' not in plain
    assert wide.pop('actor_hidden_units')==512
    assert plain==wide
