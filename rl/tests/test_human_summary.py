from summarize_human_bc import summarize_games


def test_paired_summary_uses_native_damage_not_scaled_return():
    ledger = dict(damage_dealt=400,damage_taken=1000,hits=2,blocks=1,counter_hits=1,
        combo_extensions=1,later_segment_hits=1,passive_frames=20)
    games = [dict(world_seed=seed,learner_seat=seat,outcome=outcome,frames=100,
        returns=[999,999],combat_reward_summary_by_seat=[ledger,ledger])
        for seed,seat,outcome in [(1,0,'p1_win'),(2,1,'p1_win'),(3,0,'time_limit'),(4,1,'double_ko')]]
    report = summarize_games(games)
    assert report['counts'] == dict(win=1,loss=1,time_limit=1,double_ko=1)
    assert report['win_rate'] == .25
    assert report['means']['damage_difference'] == -600
