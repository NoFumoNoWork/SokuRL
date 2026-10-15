"""Boundary timing must preserve results, exceptions and nested attribution."""
import json
from types import SimpleNamespace

import pytest

from profile_environment_interaction import Timers


def test_nested_boundary_preserves_result_and_exclusive_time(tmp_path):
    timers = Timers()
    owner = SimpleNamespace(child=lambda x: x + 1)
    owner.parent = lambda x: owner.child(x) * 2
    timers.wrap(owner, "child", "child")
    timers.wrap(owner, "parent", "parent")
    assert owner.parent(3) == 8
    timers.save(tmp_path / "timing.json")
    rows = {r["label"]: r for r in json.loads((tmp_path / "timing.json").read_text())["rows"]}
    assert rows["child"]["calls"] == rows["parent"]["calls"] == 1
    assert rows["parent"]["inclusive_seconds"] >= rows["child"]["inclusive_seconds"]
    assert rows["parent"]["exclusive_seconds"] == pytest.approx(
        rows["parent"]["inclusive_seconds"] - rows["child"]["inclusive_seconds"])


def test_exception_and_stack_cleanup():
    timers = Timers()
    error = RuntimeError("original failure")
    def fail():
        raise error
    owner = SimpleNamespace(fail=fail, ok=lambda: 7)
    timers.wrap(owner, "fail", "fail")
    timers.wrap(owner, "ok", "ok")
    with pytest.raises(RuntimeError) as caught:
        owner.fail()
    assert caught.value is error
    assert owner.ok() == 7
    assert timers.local.stack == []
    assert timers.rows["fail"]["calls"] == timers.rows["ok"]["calls"] == 1
