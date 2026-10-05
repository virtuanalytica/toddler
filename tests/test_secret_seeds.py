import stat
from datetime import date

import pytest

from toddler.learn import secret_seeds as ss
from toddler.learn import tasks as T


def test_new_set_is_private_in_band_and_matches_its_commitment(tmp_path):
    c = ss.new_set(30, date(2026, 10, 12), tmp_path)
    path = tmp_path / f"{c.set_id}.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    _, seeds, salt = ss.load_private(c.set_id, tmp_path)
    assert len(set(seeds)) == 30
    assert all(T.SECRET_SEED_LOW <= s < T.SECRET_SEED_HIGH for s in seeds)
    assert not set(seeds) & set(T.EVAL_SEEDS)
    assert max(seeds) < T.TRAIN_SEED_LOW                 # no training reset can ever draw one
    assert ss.verify(c, list(seeds), salt)


def test_commitment_detects_any_change(tmp_path):
    c = ss.new_set(5, date(2026, 10, 12), tmp_path)
    _, seeds, salt = ss.load_private(c.set_id, tmp_path)
    assert not ss.verify(c, [seeds[0] + 1, *seeds[1:]], salt)
    assert not ss.verify(c, list(seeds), salt + "0")
    assert not ss.verify(c, list(seeds[:-1]), salt)


def test_reveal_refused_before_window_closes(tmp_path):
    c = ss.new_set(3, date(2026, 10, 12), tmp_path)
    with pytest.raises(PermissionError):
        ss.reveal(c.set_id, today=date(2026, 10, 12), directory=tmp_path)
    rec = ss.reveal(c.set_id, today=date(2026, 10, 13), directory=tmp_path)
    assert ss.verify(ss.Commitment(**rec["commitment"]), rec["seeds"], rec["salt"])


def test_tampered_private_file_is_rejected(tmp_path):
    import json
    c = ss.new_set(3, date(2026, 10, 12), tmp_path)
    p = tmp_path / f"{c.set_id}.json"
    blob = json.loads(p.read_text())
    blob["seeds"][0] += 1
    p.write_text(json.dumps(blob))
    with pytest.raises(ValueError):
        ss.load_private(c.set_id, tmp_path)


@pytest.mark.parametrize("task", ["doorkey8", "unlock", "unlockpickup", "keycorridor3", "lavacross9"])
def test_harder_tasks_share_the_grid_interface(task):
    pytest.importorskip("minigrid")
    env = T.make(task, seed=10_000)
    assert env.observation_space.shape == (147,) and env.action_space.n == 7
    env.close()
