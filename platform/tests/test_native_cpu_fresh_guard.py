"""The scoped CPU reuse opt-in cannot enable arbitrary startup conditions."""
import pytest
from sokurl_platform import sokurl


@pytest.mark.parametrize('enabled,characters,cards',[(False,(1,0),True),
    (True,(0,1),True),(True,(1,0),False)])
def test_unverified_cpu_fresh_scope_rejected_before_launch(monkeypatch,enabled,characters,cards):
    monkeypatch.setenv('SOKURL_FRESH_RESET','1')
    monkeypatch.setenv('SOKURL_NATIVE_CPU_FRESH_RESET','1' if enabled else '0')
    with pytest.raises(ValueError,match='not validated'):
        sokurl._launch_vs_group_from_title(1,10,native_cpu=True,headless=True,
            characters=characters,cards_disabled=cards)


def test_verified_scope_does_not_bypass_other_startup_validation(monkeypatch):
    monkeypatch.setenv('SOKURL_FRESH_RESET','1')
    monkeypatch.setenv('SOKURL_NATIVE_CPU_FRESH_RESET','1')
    with pytest.raises(ValueError,match='stage and music'):
        sokurl._launch_vs_group_from_title(1,10,native_cpu=True,headless=True,
            characters=(1,0),cards_disabled=True,stage=-1)
