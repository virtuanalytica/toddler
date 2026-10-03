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


def test_relative_imports_from_a_package_init_resolve_to_the_package(tmp_path):
    pkg = tmp_path / "toddler"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from . import stop\n")
    (pkg / "stop.py").write_text("x = 1\n")
    (pkg / "objective.py").write_text("from .stop import x\n")
    g = cg.build(tmp_path, ("toddler",))
    assert g.edges[("logic.stop", "objective.reward")] == 1
    assert g.edges[("logic.stop", "oversight.judge")] == 1           # the __init__ re-export


def test_committed_report_matches_a_fresh_measurement():
    import json
    import subprocess
    import sys

    report = json.loads((REPO / "docs" / "design" / "module_graph.json").read_text())
    commits = [r["commit"] for r in report["revisions"]]
    for c in commits:
        if subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", c], capture_output=True).returncode:
            pytest.skip(f"commit {c} not in this clone (shallow checkout)")
    sys.path.insert(0, str(REPO / "scripts"))
    import module_graph

    fresh = [module_graph.measure(c) for c in commits]
    assert fresh == report["revisions"]
