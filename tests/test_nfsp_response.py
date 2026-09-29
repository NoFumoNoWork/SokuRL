"""Check best-response targets against the finite game's terminal payoff."""
import pytest

torch = pytest.importorskip("torch")
from soku_rl.nfsp_response import double_q_targets


def test_double_q_uses_online_selection_target_evaluation_and_terminal_payoff():
    rewards = torch.tensor([0., 0., -1.])
    final = torch.tensor([False, False, True])
    online = torch.tensor([[2., 1.], [1., 2.], [1., 1.]])
    target = torch.tensor([[.1, .9], [10., 1000.], [500., 500.]])
    legal = torch.tensor([[True, True], [True, False], [False, False]])
    bounded, raw = double_q_targets(rewards, final, online, target, legal, 1., 1.)
    torch.testing.assert_close(raw, torch.tensor([.1, 10., -1.]))
    torch.testing.assert_close(bounded, torch.tensor([.1, 1., -1.]))


def test_shaped_terminal_reward_is_not_replaced_with_the_base_payoff():
    bounded, raw = double_q_targets(torch.tensor([-1.75]), torch.tensor([True]),
        torch.zeros(1, 2), torch.ones(1, 2) * 1000, torch.ones(1, 2, dtype=torch.bool), 1., 2.)
    assert bounded.item() == raw.item() == -1.75


def test_invalid_response_targets_fail_before_clipping():
    rewards, final = torch.zeros(1), torch.tensor([False])
    values = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="legal next action"):
        double_q_targets(rewards, final, values, values, torch.zeros(1, 2, dtype=torch.bool), 1., 1.)
    with pytest.raises(ValueError, match="finite"):
        double_q_targets(rewards, final, values, values + torch.inf,
                         torch.ones(1, 2, dtype=torch.bool), 1., 1.)
