from soku_rl.rl.combo_telemetry import ComboTelemetry


def event(frame,combo,hits,segments,**extra):
    return dict(kind='HIT_RESOLUTION',attacker_seat=0,hit_or_block='HIT',
        hp_before=10000,hp_after=9800,native_combo_id=combo,combo_count_after=hits,
        combo_count_at_damage=hits-1,segment=1,frame=frame,
        independent_attack_index=segments,independent_input_segment=segments>1,
        new_independent_attack=True,counter_hit_first=False,damage_rate=.8)|extra


def info(*events,episode=1):
    return dict(learner_seat=0,episode_id=episode,native_combat_events=events)


def test_multihit_and_cross_batch_combo_are_not_three_independent_actions():
    tracker=ComboTelemetry()
    tracker.observe(0,info(event(1,1,1,1,counter_hit_first=True)),False)
    tracker.observe(0,info(event(2,1,2,1,new_independent_attack=False)),False)
    first=tracker.snapshot()
    assert first['counter_openers']==1 and first['completed_combos']==0
    tracker.reset_batch()
    tracker.observe(0,info(event(3,1,3,1,new_independent_attack=False)),False)
    tracker.observe(0,info(event(8,2,1,1)),False)
    second=tracker.snapshot()
    assert second['counter_openers']==0 and second['combo_extensions']==0
    assert second['native_hit_length_histogram']=={3:1}
    assert second['independent_segment_lower_bound_histogram']=={1:1}
    assert second['closed_combos'][0]['counter_opened']


def test_aaa_and_unknown_have_distinct_lower_bound_statistics():
    tracker=ComboTelemetry()
    for n in (1,2,3):tracker.observe(0,info(event(n,1,n,n)),n==3)
    value=tracker.snapshot()
    assert value['combo_extensions']==2
    assert value['native_hit_length_histogram']=={3:1}
    assert value['independent_segment_lower_bound_histogram']=={3:1}
    tracker.reset_batch()
    tracker.observe(0,info(event(4,1,1,1),episode=2),False)
    tracker.observe(0,info(event(5,1,2,1,independent_attack_index=None,
                    independent_input_segment=None),episode=2),True)
    value=tracker.snapshot()
    assert value['completed_with_unknown_lineage']==1
    assert value['independent_segment_lower_bound_histogram']=={1:1}


def test_missing_attribution_does_not_discard_proven_counter():
    tracker=ComboTelemetry()
    tracker.observe(0,info(event(1,None,1,1,counter_hit_first=True)),True)
    value=tracker.snapshot()
    assert value['counter_openers']==1 and value['counter_damage']==200
    assert value['unresolved_hits']==1 and value['completed_combos']==0


def test_mid_combo_capture_is_censored_not_completed():
    tracker=ComboTelemetry()
    tracker.observe(0,info(event(10,1,4,2)),True)
    value=tracker.snapshot()
    assert value['completed_combos']==0
    assert value['closed_combos'][0]['left_censored']


def test_unknown_hit_keeps_action_context_not_just_count():
    tracker=ComboTelemetry()
    source=dict(kind='ACTION_TRANSITION',attacker_seat=0,frame=1,order=1,
                attack_instance=5,accepted=True,native_input_scope=None,caller=0x1234)
    hit=event(2,1,1,1,attack_instance=5,independent_input_segment=None,
              independent_segment_unknown_reason='ACTION_NOT_PROVEN_AS_INPUT_ACCEPTED')
    tracker.observe(0,info(source,hit),True)
    record=tracker.snapshot()['unknown_events'][0]
    assert record['event']['independent_segment_unknown_reason']=='ACTION_NOT_PROVEN_AS_INPUT_ACCEPTED'
    assert record['source_context'][0]['caller']==0x1234


def test_taken_counter_is_not_agent_output_counter():
    tracker=ComboTelemetry()
    tracker.observe(0,info(event(1,1,1,1,attacker_seat=1,counter_hit_first=True)),False)
    value=tracker.snapshot()
    assert value['counter_taken']==1 and value['counter_openers']==0
    assert value['hits_taken']==1 and value['hits']==0
