from types import SimpleNamespace
from soku_rl.env.passive import PassiveQualifier


def raw(frame,x=0,action=0,**kwargs):
    p=lambda position:SimpleNamespace(x=position,actionId=action,frameFlags=0,hitstop=0,untech=0,timeStop=0,**kwargs)
    return SimpleNamespace(frameId=frame,p1=p(x),p2=p(800))


def test_native_forced_states_and_real_pursuit_not_whiff_commands():
    classifier=PassiveQualifier()
    classifier.observe(raw(0),())
    assert classifier.observe(raw(1),())['qualified_passive']==(True,True)
    assert classifier.observe(raw(2,x=5),())['qualified_passive']==(False,True)
    assert classifier.observe(raw(3,x=5,action=97),())['qualified_passive']==(False,False)
    # A whiff is not automatically considered productive interaction.
    assert classifier.observe(raw(4,x=5,action=300),())['qualified_passive']==(True,True)
    contact=dict(kind='BLOCK_HIGH_RESOLUTION',hit_or_block='BLOCK')
    assert classifier.observe(raw(5,x=5),(contact,))['passive_reset']==(True,True)
