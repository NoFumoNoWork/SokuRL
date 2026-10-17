"""Do not turn horizon HP leads into native victories."""
import pytest
from evaluate_event_duel import duel_result


@pytest.mark.parametrize('seat', [0, 1])
def test_native_winner_is_relative_to_bc_seat(seat):
    assert duel_result(f'p{seat+1}_win', seat, [0, 0]) == 'BC_KO_WIN'
    assert duel_result(f'p{2-seat}_win', seat, [0, 0]) == 'PPO_KO_WIN'


def test_timeout_lead_is_not_ko():
    assert duel_result('time_limit', 0, [9000, 5000]) == 'TIMEOUT_BC_HP_LEAD'
    assert duel_result('time_limit', 1, [9000, 5000]) == 'TIMEOUT_PPO_HP_LEAD'
    assert duel_result('time_limit', 0, [5000, 5000]) == 'TIMEOUT_HP_TIE'
    assert duel_result('double_ko', 1, [0, 0]) == 'DOUBLE_KO'
    with pytest.raises(ValueError):duel_result('ongoing', 0, [9000, 5000])
