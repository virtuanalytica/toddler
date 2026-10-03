"""Measure Toddler's own module graph over the ATLAS for one or more git revisions (step B).

Run: PYTHONPATH=. python3 scripts/module_graph.py [REV ...]   (default: HEAD)
Report: docs/design/module_graph.json
Each revision is exported with `git archive` into a temporary directory, so the working tree is
never touched and every number comes from committed code.
"""

import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from toddler import codegraph as cg
from toddler.structure import ATLAS, clustering_profile

REPO = Path(__file__).resolve().parents[1]


def measure(rev: str) -> dict:
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", rev], capture_output=True,
                         text=True, check=True).stdout.strip()
    paths = [p for p in ("toddler", "jevserver")
             if subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{rev}:{p}"], capture_output=True).returncode == 0]
    blob = subprocess.run(["git", "-C", str(REPO), "archive", rev, *paths], capture_output=True, check=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        tarfile.open(fileobj=io.BytesIO(blob)).extractall(tmp, filter="data")
        pkgs = tuple(p for p in ("toddler", "jevserver") if (Path(tmp) / p).exists())
        g = cg.build(Path(tmp), pkgs)
    norm = cg.normalised(g)
    return {"rev": rev, "commit": sha,
            "edges": [{"a": a, "b": b, "imports": w} for (a, b), w in sorted(g.edges.items())],
            "sizes": g.sizes,
            "normalised": {f"{ATLAS[i]}|{ATLAS[j]}": round(float(norm[i, j]), 6)
                           for i in range(len(ATLAS)) for j in range(i + 1, len(ATLAS)) if norm[i, j] > 0},
            "clustering": dict(zip(ATLAS, [round(float(x), 4) for x in clustering_profile(norm)])),
            "empty_regions": [r for r in ATLAS if g.sizes[r] == 0]}


def main() -> None:
    revs = sys.argv[1:] or ["HEAD"]
    report = {"atlas": list(ATLAS), "method": "static imports between regions (ast), sizes = non-blank lines",
              "revisions": [measure(r) for r in revs]}
    out = REPO / "docs" / "design" / "module_graph.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    for r in report["revisions"]:
        print(r["commit"], len(r["edges"]), "region pairs, empty:", r["empty_regions"])


if __name__ == "__main__":
    main()
