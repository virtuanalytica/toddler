"""Mapping step B: turn Toddler's own code into a module graph over the ATLAS.

Every Python module of the repository is assigned to exactly one ATLAS region (MODULE_REGION).
An edge between two regions counts the static imports from modules of one region to modules of
the other (parsed with `ast`, nothing is executed); a region's size is its number of non-blank
source lines. The result feeds `structure.align_to_atlas` / `normalise_by_size` /
`clustering_profile`, so Toddler versions (git revisions) can be compared on the same regions.

Regions without code yet (perception, actuation) stay empty: an absent capability shows as
zero, never as an invented edge. A module that is not in MODULE_REGION is an error, so a new
module must be placed on the atlas explicitly before the graph can be measured.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toddler.structure import ATLAS, align_to_atlas, normalise_by_size

# module (dotted, without .py) -> ATLAS region; packages map by prefix ("toddler.learn")
MODULE_REGION: dict[str, str] = {
    "toddler.stop": "logic.stop",
    "toddler.specialize": "logic.allowlist",
    "toddler.config": "logic.allowlist",
    "toddler.credentials": "logic.allowlist",   # access control: identities, governed keys, vault
    "toddler.provenance": "logic.provenance",
    "toddler.fastpath": "fastpath.jev",
    "toddler.jev": "fastpath.jev",
    "toddler.clm": "fastpath.jev",
    "toddler.reflex_shadow": "fastpath.jev",
    "jevserver": "fastpath.jev",
    "toddler.g4_curriculum": "planner",
    "toddler.g4_student": "planner",
    "toddler.model_mixture_search": "planner",
    "toddler.learn": "planner",                # the learning brain: policies, PPO, generations
    "toddler.knowledge.publish": "p2p.knitweb",
    "toddler.knowledge": "memory.graph",
    "toddler.objective": "objective.reward",
    "toddler.business": "objective.profit",
    "toddler.resources": "objective.energy",
    "toddler._knitweb": "p2p.knitweb",
    "toddler.relay": "p2p.pulse_relay",
    "toddler.evaluation": "oversight.judge",
    "toddler.quotients": "oversight.judge",
    "toddler.structure": "oversight.judge",
    "toddler.codegraph": "oversight.judge",     # this module: measurement, like structure
    "toddler.audit": "oversight.audit",
    "toddler.selfheal": "oversight.audit",      # crash watchdog: classify, retry, quarantine
    "toddler": "oversight.judge",              # toddler/__init__.py (package marker only)
}
EXACT_ONLY = {"toddler"}    # never a prefix: a new toddler.* module must be placed explicitly


def module_name(root: Path, path: Path) -> str:
    """Dotted name; a package's __init__ is the package itself. __main__ keeps its name on
    purpose (jevserver.__main__ is placed through the "jevserver" prefix; a future
    toddler/__main__.py would need its own MODULE_REGION entry, since "toddler" is exact-only)."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def region_of(module: str) -> str:
    """Longest matching prefix in MODULE_REGION."""
    best = ""
    for key in MODULE_REGION:
        prefix_ok = key not in EXACT_ONLY and module.startswith(key + ".")
        if (module == key or prefix_ok) and len(key) > len(best):
            best = key
    if not best:
        raise KeyError(f"module {module!r} is not placed on the atlas (add it to MODULE_REGION)")
    region = MODULE_REGION[best]
    if region not in ATLAS:
        raise KeyError(f"region {region!r} of {module!r} is not in the ATLAS")
    return region


def _imports(tree: ast.AST, module: str, is_package: bool = False) -> list[str]:
    out = []
    package = module.split(".") if is_package else module.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative import: resolve against the package
                base = package[: len(package) - (node.level - 1)]
                prefix = ".".join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            out += [f"{prefix}.{a.name}" if prefix else a.name for a in node.names]
    return out


@dataclass(frozen=True)
class CodeGraph:
    edges: dict[tuple[str, str], float]      # (region, region) -> number of imports, a < b
    sizes: dict[str, int]                    # region -> non-blank source lines
    modules: dict[str, str]                  # module -> region


def build(root: Path, packages: tuple[str, ...] = ("toddler", "jevserver")) -> CodeGraph:
    root = Path(root)
    modules, sizes, edges = {}, Counter(), Counter()
    trees, packages_seen = {}, set()
    for pkg in packages:
        for path in sorted((root / pkg).rglob("*.py")):
            mod = module_name(root, path)
            src = path.read_text(encoding="utf-8")
            modules[mod] = region_of(mod)
            sizes[modules[mod]] += sum(1 for line in src.splitlines() if line.strip())
            trees[mod] = ast.parse(src, filename=str(path))
            if path.name == "__init__.py":
                packages_seen.add(mod)
    for mod, tree in trees.items():
        for target in _imports(tree, mod, mod in packages_seen):
            if not target.split(".")[0] in packages:
                continue                          # third-party or stdlib
            b, a = region_of(target), modules[mod]       # "toddler.stop.allowed" -> prefix toddler.stop
            if a != b:
                edges[tuple(sorted((a, b)))] += 1
    return CodeGraph(dict(edges), {r: sizes.get(r, 0) for r in ATLAS}, modules)


def normalised(g: CodeGraph) -> np.ndarray:
    """Row 8 normalisation over the regions that have code; empty regions stay 0 (an absent
    capability has no size to divide by and no edges to report)."""
    present = [r for r in ATLAS if g.sizes[r] > 0]
    sub = normalise_by_size(align_to_atlas(g.edges, present), [g.sizes[r] for r in present])
    full = np.zeros((len(ATLAS), len(ATLAS)))
    idx = [ATLAS.index(r) for r in present]
    full[np.ix_(idx, idx)] = sub
    return full
