from pathlib import Path

import numpy as np
import pytest

from toddler import codegraph as cg
from toddler.structure import ATLAS

REPO = Path(__file__).resolve().parents[1]


def test_every_module_of_the_repository_is_on_the_atlas():
    g = cg.build(REPO)
    assert g.modules and set(g.modules.values()) <= set(ATLAS)


def test_real_imports_become_region_edges():
    g = cg.build(REPO)
    assert g.edges[("logic.stop", "objective.reward")] >= 1        # objective.py imports toddler.stop
    assert all(a < b for a, b in g.edges)


def test_unknown_module_is_refused(tmp_path):
    (tmp_path / "toddler").mkdir()
    (tmp_path / "toddler" / "teleport.py").write_text("x = 1\n")
    (tmp_path / "toddler" / "__init__.py").write_text("")
    with pytest.raises(KeyError, match="not placed on the atlas"):
        cg.build(tmp_path, ("toddler",))


def test_empty_regions_stay_zero_after_normalisation():
    g = cg.build(REPO)
    n = cg.normalised(g)
    for r in ("perception.vision", "actuation"):
        i = ATLAS.index(r)
        assert g.sizes[r] == 0 and not n[i].any() and not n[:, i].any()
    assert np.allclose(n, n.T)
