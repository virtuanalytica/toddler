"""Structural analysis of Toddler's module graph, following Wee et al. (2017).

Each function implements one step of the paper's pipeline, applied to Toddler's own
code/capability graph instead of an infant brain (docs/design/wee2017-mapping.md):

  row 6   shared atlas            -> ATLAS / align_to_atlas
  row 8   volume normalisation    -> normalise_by_size
          noise-edge sign test    -> sign_test_edges
  row 9   clustering coefficient  -> clustering_profile
  row 10  subject similarity      -> similarity_matrix / trim_threshold
  row 11  consensus clustering    -> consensus_partition
  row 12  null model              -> modularity_null_test
  row 13  assignment confidence   -> assignment_confidence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
from scipy import stats

# Row 6: a fixed capability ontology so that every Toddler version is measured on the same
# "regions". Order matters: profile vectors are indexed by this tuple.
ATLAS: tuple[str, ...] = (
    "logic.stop",
    "logic.allowlist",
    "logic.provenance",
    "perception.web",
    "perception.vision",
    "perception.lidar",
    "fastpath.jev",
    "planner",
    "memory.graph",
    "objective.reward",
    "objective.profit",
    "objective.energy",
    "p2p.knitweb",
    "p2p.pulse_relay",
    "actuation",
    "oversight.judge",
    "oversight.audit",
)

Edges = Mapping[tuple[str, str], float]


def align_to_atlas(edges: Edges, atlas: Sequence[str] = ATLAS) -> np.ndarray:
    """Weighted adjacency matrix over the atlas; edges to unknown modules are rejected."""
    index = {name: i for i, name in enumerate(atlas)}
    m = np.zeros((len(atlas), len(atlas)))
    for (a, b), w in edges.items():
        if a not in index or b not in index:
            raise KeyError(f"module not in atlas: {a if a not in index else b}")
        if a == b:
            continue
        m[index[a], index[b]] += w
        m[index[b], index[a]] += w
    return m


def normalise_by_size(adj: np.ndarray, sizes: Sequence[float]) -> np.ndarray:
    """Row 8: connection count divided by the mean size of the two modules (paper: mean region volume)."""
    s = np.asarray(sizes, dtype=float)
    if s.shape != (adj.shape[0],) or np.any(s <= 0):
        raise ValueError("sizes must be positive, one per atlas module")
    return adj / ((s[:, None] + s[None, :]) / 2.0)


def sign_test_edges(runs: Sequence[np.ndarray], alpha: float = 0.05) -> np.ndarray:
    """Row 8: keep an edge only if it is present (>0) in significantly more than half of the runs.

    One-tailed sign test across runs, as the paper does across subjects to remove
    tractography noise. Returns a boolean mask.
    """
    stack = np.stack(runs)
    n = stack.shape[0]
    positive = (stack > 0).sum(axis=0)
    p = stats.binom.sf(positive - 1, n, 0.5)  # P(X >= positive) under H0: p = 0.5
    mask = p < alpha
    np.fill_diagonal(mask, False)
    return mask


def clustering_profile(adj: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Row 9: binary clustering coefficient per module (paper: Brain Connectivity Toolbox, binary)."""
    binary = adj > 0
    if mask is not None:
        binary &= mask
    g = nx.from_numpy_array(binary.astype(int))
    cc = nx.clustering(g)
    return np.array([cc[i] for i in range(adj.shape[0])])


def standardise(profile: np.ndarray) -> np.ndarray:
    """Profiles are z-scored across modules before comparison, as in the paper."""
    sd = profile.std()
    return (profile - profile.mean()) / sd if sd > 0 else np.zeros_like(profile)


def similarity_matrix(profiles: Sequence[np.ndarray]) -> np.ndarray:
    """Row 10: Pearson correlation between every pair of versions' standardised profiles."""
    z = np.stack([standardise(p) for p in profiles])
    sim = np.corrcoef(z)
    return np.nan_to_num(sim)


def trim_threshold(sim: np.ndarray) -> float:
    """Row 10: the largest threshold that still leaves every node with at least one edge.

    The paper picked r = 0.60 this way: trim weak edges but keep the graph free of isolates.
    """
    off = sim.copy()
    np.fill_diagonal(off, -np.inf)
    return float(off.max(axis=1).min())


def _graph_from_similarity(sim: np.ndarray, threshold: float) -> nx.Graph:
    g = nx.Graph()
    n = sim.shape[0]
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                g.add_edge(i, j, weight=float(sim[i, j]))
    return g


@dataclass(frozen=True)
class ConsensusResult:
    labels: np.ndarray
    co_assignment: np.ndarray
    mean_modularity: float


def consensus_partition(sim: np.ndarray, threshold: float, runs: int = 1000,
                        consensus_cut: float = 0.2, seed: int = 0) -> ConsensusResult:
    """Row 11: Louvain is non-deterministic, so run it many times, build the co-assignment
    (consensus) matrix, drop pairs below `consensus_cut`, and cluster the consensus graph.
    The paper used 10,000 runs and cut values 0.20-0.35."""
    g = _graph_from_similarity(sim, threshold)
    n = sim.shape[0]
    co = np.zeros((n, n))
    qs = []
    rng = np.random.default_rng(seed)
    for _ in range(runs):
        parts = nx.community.louvain_communities(g, weight="weight", seed=int(rng.integers(2**31)))
        qs.append(nx.community.modularity(g, parts, weight="weight"))
        for part in parts:
            idx = np.fromiter(part, dtype=int)
            co[np.ix_(idx, idx)] += 1
    co /= runs
    cg = nx.Graph()
    cg.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if co[i, j] >= consensus_cut:
                cg.add_edge(i, j, weight=float(co[i, j]))
    final = nx.community.louvain_communities(cg, weight="weight", seed=seed)
    labels = np.empty(n, dtype=int)
    for k, part in enumerate(sorted(final, key=min)):
        labels[list(part)] = k
    return ConsensusResult(labels=labels, co_assignment=co, mean_modularity=float(np.mean(qs)))


@dataclass(frozen=True)
class NullTest:
    observed_q: float
    null_mean: float
    null_sd: float
    z: float
    p_value: float


def modularity_null_test(sim: np.ndarray, threshold: float, observed_q: float,
                         n_null: int = 1000, seed: int = 0) -> NullTest:
    """Row 12: compare the observed modularity with graphs that keep the node count, edge count
    and edge weights but rewire the structure (paper: 10,000 random graphs, Q = 0.293, z = 45)."""
    g = _graph_from_similarity(sim, threshold)
    weights = [d["weight"] for _, _, d in g.edges(data=True)]
    rng = np.random.default_rng(seed)
    null_qs = []
    for _ in range(n_null):
        r = nx.gnm_random_graph(g.number_of_nodes(), g.number_of_edges(), seed=int(rng.integers(2**31)))
        for (u, v), w in zip(r.edges(), rng.permutation(weights)):
            r[u][v]["weight"] = float(w)
        parts = nx.community.louvain_communities(r, weight="weight", seed=int(rng.integers(2**31)))
        null_qs.append(nx.community.modularity(r, parts, weight="weight"))
    null = np.asarray(null_qs)
    sd = float(null.std()) or 1e-12
    p = (np.sum(null >= observed_q) + 1) / (n_null + 1)
    return NullTest(observed_q, float(null.mean()), sd, (observed_q - float(null.mean())) / sd, float(p))


def assignment_confidence(result: ConsensusResult) -> np.ndarray:
    """Row 13: per node, the mean co-assignment with the other members of its final group.
    The paper reports the share of subjects above 0.75; below 0.5 means 'ask a human'."""
    n = len(result.labels)
    conf = np.ones(n)
    for i in range(n):
        mates = [j for j in range(n) if j != i and result.labels[j] == result.labels[i]]
        if mates:
            conf[i] = float(result.co_assignment[i, mates].mean())
    return conf
