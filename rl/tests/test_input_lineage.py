from soku_rl.env.input_lineage import InputLineage


def scope(frame, button=None):
    keys=[0]*8
    if button is not None:keys[button]=1
    return dict(segment=1,frame=frame,kind='NATIVE_INPUT_SCOPE',attacker_seat=0,
                native_input_scope=frame,native_inputs_before=keys)


def action(frame,root,move,button=2):
    keys=[0]*8;keys[button]=1
    return dict(segment=1,frame=frame,kind='ACTION_TRANSITION',attacker_seat=0,
        native_input_scope=frame,action_after=move,attack_instance=root,accepted=True,
        lineage='DIRECT',native_inputs_before=keys,buffered_inputs_before=[0]*8,
        released_inputs_before=[0]*6)


def contact(frame,root,count):
    return dict(segment=1,frame=frame,kind='HIT_RESOLUTION',attacker_seat=0,
        hit_or_block='HIT',attack_instance=root,combo_count_at_damage=count,
        independent_input_segment=None,hp_before=10000,hp_after=9900,lineage='DIRECT')


def test_accepted_root_unknown_reasons_do_not_relabel_damage_or_combo():
    tracker=InputLineage(accepted_action_roots=True)
    tracker.observe([action(1,10,400),contact(1,10,0)])
    auto=action(2,20,401);auto['native_input_scope']=None
    result=tracker.observe([auto,contact(2,20,1)|{'counter_hit_first':False}])[-1]
    assert result['independent_segment_unknown_reason']=='ACTION_NOT_PROVEN_AS_INPUT_ACCEPTED'
    assert result['native_combo_id']==1 and result['hp_after']==9900
    assert result['counter_hit_first'] is False
    result=tracker.observe([contact(3,None,2)|{'lineage':'UNKNOWN'}])[-1]
    assert result['independent_segment_unknown_reason']=='UNPROVEN_SPAWN_PROVENANCE'


def test_aaa_separate_presses_and_multihit_stay_one_input_segment():
    tracker=InputLineage(diagnostic_action_mapping=True)
    first=tracker.observe([scope(1,2),action(1,10,300),contact(1,10,0)])[-1]
    assert first['independent_attack_index']==1
    tracker.observe([scope(2)])
    second=tracker.observe([scope(3,2),action(3,20,320),contact(3,20,1)])[-1]
    assert second['independent_attack_index']==2
    assert second['independent_input_segment'] is True
    repeated=tracker.observe([scope(4,2),contact(4,20,2)])[-1]
    assert repeated['independent_attack_index']==2
    tracker.observe([scope(5)])
    third=tracker.observe([scope(6,2),action(6,30,321),contact(6,30,3)])[-1]
    assert third['independent_attack_index']==3


def test_hold_automatic_action_whiff_and_unknown_skill_do_not_invent_segments():
    tracker=InputLineage(diagnostic_action_mapping=True)
    tracker.observe([scope(1,2),action(1,10,300),contact(1,10,0)])
    next_hit=tracker.observe([scope(2,2),action(2,20,320),contact(2,20,1)])[-1]
    assert next_hit['independent_attack_index']==1
    assert next_hit['independent_input_segment'] is False
    tracker.observe([scope(3),scope(4,2),action(4,30,321)])
    assert len(tracker.combo[0])==1  # accepted but whiffed
    tracker.observe([scope(5),scope(6,3),action(6,40,500,3)])
    assert tracker.observe([contact(7,40,2)])[-1]['independent_input_segment'] is None


def test_missing_input_frames_and_outside_input_scope_remain_unknown():
    tracker=InputLineage(diagnostic_action_mapping=True)
    tracker.observe([scope(1,2),action(1,10,300),contact(1,10,0)])
    assert tracker.observe([scope(8,2),action(8,20,320),contact(8,20,1)])[-1]['independent_input_segment'] is None
    other=action(9,30,321);other['native_input_scope']=None
    assert tracker.observe([other,contact(9,30,2)])[-1]['independent_input_segment'] is None


def test_old_buffer_and_new_press_are_not_silently_assigned_to_new_input():
    tracker=InputLineage(diagnostic_action_mapping=True)
    first=scope(1,2);first['buffered_inputs_before']=[0,0,1,0,0,0,0,0]
    start=action(1,10,300);start['buffered_inputs_before']=[0,0,1,0,0,0,0,0]
    tracker.observe([first,start,contact(1,10,0)])
    release=scope(2);release['buffered_inputs_before']=[0,0,1,0,0,0,0,0]
    tracker.observe([release])
    new=scope(3,2);new['buffered_inputs_before']=[0,0,1,0,0,0,0,0]
    ambiguous=action(3,20,320);ambiguous['buffered_inputs_before']=[0,0,1,0,0,0,0,0]
    assert tracker.observe([new,ambiguous,contact(3,20,1)])[-1]['independent_input_segment'] is None


def test_hitstop_frame_samples_preserve_buffered_press_and_default_skill_mapping():
    tracker=InputLineage(diagnostic_action_mapping=True)
    tracker.observe([scope(1,2),action(1,10,300),contact(1,10,0)],characters=(0,1))
    tracker.observe([scope(2)],characters=(0,1))
    sample=scope(3,4);sample['kind']='NATIVE_INPUT_SAMPLE'
    sample['buffered_inputs_before']=[0,0,0,0,1,0,0,0]
    tracker.observe([sample],characters=(0,1))
    sample=scope(4);sample['kind']='NATIVE_INPUT_SAMPLE'
    sample['buffered_inputs_before']=[0,0,0,0,1,0,0,0]
    tracker.observe([sample],characters=(0,1))
    buffered=scope(5);buffered['buffered_inputs_before']=[0,0,0,0,1,0,0,0]
    accepted=action(5,20,410,4);accepted['native_inputs_before']=[0]*8
    accepted['buffered_inputs_before']=[0,0,0,0,1,0,0,0]
    event=tracker.observe([buffered,accepted,contact(5,20,1)],characters=(0,1))[-1]
    assert event['independent_attack_index']==2
    tracker.observe([scope(6)],characters=(0,1))
    event=tracker.observe([scope(7,3),action(7,30,560,3),contact(7,30,2)],characters=(0,1))[-1]
    assert event['independent_attack_index']==3


def test_two_unused_buffer_candidates_prove_one_new_segment_not_two():
    tracker=InputLineage(diagnostic_action_mapping=True)
    tracker.observe([scope(1,2),action(1,10,300),contact(1,10,0)])
    tracker.observe([scope(2)])
    for frame,button in ((3,4),(4,None),(5,4),(6,None)):
        sample=scope(frame,button);sample['kind']='NATIVE_INPUT_SAMPLE'
        sample['buffered_inputs_before']=[0,0,0,0,1,0,0,0]
        tracker.observe([sample])
    buffered=scope(7);buffered['buffered_inputs_before']=[0,0,0,0,1,0,0,0]
    accepted=action(7,20,410,4);accepted['native_inputs_before']=[0]*8
    accepted['buffered_inputs_before']=[0,0,0,0,1,0,0,0]
    hit=tracker.observe([buffered,accepted,contact(7,20,1)])[-1]
    assert hit['independent_input_segment'] is True
    assert hit['independent_attack_index']==2
    assert hit['independent_input_id'] is None
    assert len(hit['input_candidates'])==2
    repeated=tracker.observe([contact(8,20,2)])[-1]
    assert repeated['independent_attack_index']==2


def test_formal_ledger_ignores_action_mapping_and_accepts_arbitrary_action_with_evidence():
    tracker=InputLineage()
    tracker.observe([scope(1,2),action(1,10,300),contact(1,10,0)])
    result=tracker.observe([scope(2),scope(3,2),action(3,20,320),contact(3,20,1)])[-1]
    assert result['independent_input_segment'] is None
    tracker=InputLineage()
    first=action(1,10,1999)
    first.update(accepted_input_candidates=[1],input_acceptance_evidence='NATIVE_INPUT_CONSUMPTION')
    tracker.observe([first,contact(1,10,0)])
    second=action(2,20,1777)
    second.update(accepted_input_candidates=[2],input_acceptance_evidence='NATIVE_INPUT_CONSUMPTION')
    result=tracker.observe([second,contact(2,20,1)])[-1]
    assert result['independent_input_segment'] is True
    assert result['independent_attack_index']==2


def test_generic_native_scope_decisions_do_not_require_move_ids():
    tracker=InputLineage(accepted_scope_decisions=True)
    first=tracker.observe([scope(1,2),action(1,10,1999),contact(1,10,0)])[-1]
    tracker.observe([scope(2)])
    second=tracker.observe([scope(3,2),action(3,20,1777),contact(3,20,1)])[-1]
    assert second['independent_attack_index']==2
    assert second['input_lineage_evidence']=='NATIVE_ACCEPTED_SCOPE_DECISION_CANDIDATES'
    repeated=tracker.observe([scope(4,2),contact(4,20,2)])[-1]
    assert repeated['independent_attack_index']==2
    reset=tracker.observe([scope(5),scope(6,3),action(6,30,1666,3),contact(6,30,0)])[-1]
    assert reset['independent_attack_index']==1 and reset['independent_input_segment'] is False


def test_generic_scope_old_buffer_overlap_remains_unknown():
    tracker=InputLineage(accepted_scope_decisions=True)
    first=scope(1,2);first['buffered_inputs_before']=[0,0,1,0,0,0,0,0]
    accepted=action(1,10,1999);accepted['buffered_inputs_before']=first['buffered_inputs_before']
    tracker.observe([first,accepted,contact(1,10,0)])
    release=scope(2);release['buffered_inputs_before']=first['buffered_inputs_before']
    tracker.observe([release])
    pressed=scope(3,2);pressed['buffered_inputs_before']=first['buffered_inputs_before']
    accepted=action(3,20,1777);accepted['buffered_inputs_before']=first['buffered_inputs_before']
    event=tracker.observe([pressed,accepted,contact(3,20,1)])[-1]
    assert event['input_candidates']==(1,2)
    assert event['independent_input_segment'] is None
    assert event['new_independent_attack'] is False


def test_simultaneous_buttons_are_one_decision_not_multiple_segments():
    tracker=InputLineage(accepted_scope_decisions=True)
    pressed=scope(1,2);pressed['native_inputs_before'][3]=1
    accepted=action(1,10,1999);accepted['native_inputs_before'][3]=1
    event=tracker.observe([pressed,accepted,contact(1,10,0)])[-1]
    assert event['input_candidates']==(1,)
    same=tracker.observe([contact(2,10,1)])[-1]
    assert same['independent_attack_index']==1
    assert same['new_independent_attack'] is False


def test_accepted_action_roots_need_no_key_or_buffer_reconstruction():
    tracker=InputLineage(accepted_action_roots=True)
    for frame,root,move,count,index in ((1,10,1999,0,1),(2,20,1777,1,2),(3,30,1777,2,3)):
        accepted=action(frame,root,move)
        for field in ('native_inputs_before','buffered_inputs_before','released_inputs_before'):
            del accepted[field]
        event=tracker.observe([accepted,contact(frame,root,count)])[-1]
        assert event['independent_attack_index']==index
        assert event['independent_action_instance']==root
        assert event['independent_input_id'] is None
        assert event['input_lineage_evidence']=='NATIVE_INPUT_ACCEPTED_ACTION_INSTANCE'
    # A repeated collision, second projectile or child inherits the same root.
    child=contact(4,10,3);child.update(lineage='CHILD',source_instance=900)
    event=tracker.observe([child])[-1]
    assert event['independent_attack_index']==1
    assert event['new_independent_attack'] is False
    repeated=tracker.observe([contact(5,20,4)])[-1]
    assert repeated['independent_attack_index']==2
    assert repeated['new_independent_attack'] is False
    # Resolver-internal reset starts a new combo even for an old projectile.
    reset_contact=contact(6,10,0)
    reset_contact.update(combo_count_before=6,combo_count_after=1)
    reset=tracker.observe([reset_contact])[-1]
    assert reset['native_combo_id']==2 and reset['independent_attack_index']==1


def test_automatic_failed_and_unknown_roots_do_not_invent_operations():
    tracker=InputLineage(accepted_action_roots=True)
    tracker.observe([action(1,10,1999),contact(1,10,0)])
    automatic=action(2,20,1777);automatic['native_input_scope']=None
    event=tracker.observe([automatic,contact(2,20,1)])[-1]
    assert event['independent_input_segment'] is None
    assert event['combo_evidence']=='NATIVE_DAMAGE_TIME_COMBO_COUNT'
    assert event['attack_root_evidence']=='NATIVE_ACTION_SPAWN_ROOT'
    rejected=action(3,30,1666);rejected['accepted']=False
    assert tracker.observe([rejected,contact(3,30,2)])[-1]['independent_input_segment'] is None
    missing=contact(4,10,3);missing['lineage']='UNKNOWN'
    assert tracker.observe([missing])[-1]['independent_input_segment'] is None
