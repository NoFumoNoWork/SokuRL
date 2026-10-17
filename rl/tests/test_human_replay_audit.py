import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('human_audit', Path(__file__).parents[1] / 'tools/audit_human_replays.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def row(n, focal, opponent):
    return dict(id=n, set_id=str(n), players=dict(server=dict(id=focal), client=dict(id=opponent)),
                sampling=dict(focal_side='server', month='2026-01'))


def test_split_does_not_share_players_or_sets():
    rows = [row(n, n, n+100) for n in range(20)]
    rows += [row(100+n, n, n+1) for n in range(19)]
    train, validation, excluded = audit.split_training(rows, 123)
    assert train and validation
    assert not set().union(*(audit.players(r) for r in train)) & set().union(*(audit.players(r) for r in validation))
    assert len(train) + len(validation) + len(excluded) == len(rows)
    assert audit.split_training(rows, 123) == (train, validation, excluded)


def test_pilot_does_not_reuse_identity():
    rows = [row(n, n, n+100) for n in range(30)]
    chosen = audit.choose_pilot(rows)
    assert len(chosen) == len({r['id'] for r in chosen}) == 20
    assert rows == [row(n, n, n+100) for n in range(30)]


def test_preserve_previous_output(tmp_path):
    path = tmp_path / 'old.json'
    audit.write_new(path, {'a': 1})
    with pytest.raises(FileExistsError):
        audit.write_new(path, {})
    assert path.read_text().strip().startswith('{')
