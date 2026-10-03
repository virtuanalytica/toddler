"""Tests use Zachary's karate club (real observed social network, shipped with networkx)
for the clustering steps, and small explicit matrices for the arithmetic steps."""

import networkx as nx
import numpy as np
import pytest

from toddler import structure as s


def _karate_sim():
    g = nx.karate_club_graph()
    adj = nx.to_numpy_array(g, weight=None)
    np.fill_diagonal(adj, 1.0)
    return adj


def test_align_rejects_unknown_module():
    with pytest.raises(KeyError):
        s.align_to_atlas({("planner", "not.a.module"): 1})


def test_align_is_symmetric_and_ignores_self_loops():
    m = s.align_to_atlas({("planner", "logic.stop"): 3, ("planner", "planner"): 9})
    i, j = s.ATLAS.index("planner"), s.ATLAS.index("logic.stop")
    assert m[i, j] == m[j, i] == 3
    assert m[i, i] == 0


def test_normalise_by_mean_size():
    adj = np.array([[0, 4.0], [4.0, 0]])
    out = s.normalise_by_size(adj, [2, 6])
    assert out[0, 1] == pytest.approx(1.0)


def test_sign_test_keeps_consistent_edges_only():
    consistent = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])
    noisy = np.array([[0, 1.0, 1.0], [1.0, 0, 0], [1.0, 0, 0]])
    runs = [consistent] * 9 + [noisy]
    mask = s.sign_test_edges(runs)
    assert mask[0, 1] and not mask[0, 2]


def test_clustering_profile_triangle():
    tri = np.ones((3, 3)) - np.eye(3)
    assert s.clustering_profile(tri).tolist() == [1.0, 1.0, 1.0]


def test_trim_threshold_leaves_no_isolates():
    sim = s.similarity_matrix([np.array([1, 2, 3.0]), np.array([1, 2, 3.1]), np.array([3, 1, 2.0])])
    t = s.trim_threshold(sim)
    g = s._graph_from_similarity(sim, t)
    assert min(dict(g.degree()).values()) >= 1


def test_consensus_recovers_karate_factions():
    sim = _karate_sim()
    res = s.consensus_partition(sim, threshold=0.5, runs=200, seed=1)
    truth = np.array([nx.karate_club_graph().nodes[i]["club"] == "Mr. Hi" for i in range(34)])
    # Every consensus group is dominated by one faction (purity), the paper's 'meaningful groups'.
    purity = sum(max(np.sum(truth[res.labels == k]), np.sum(~truth[res.labels == k]))
                 for k in np.unique(res.labels)) / 34
    assert purity >= 0.85


def test_null_model_rejects_random_structure_for_karate():
    sim = _karate_sim()
    res = s.consensus_partition(sim, threshold=0.5, runs=50, seed=2)
    null = s.modularity_null_test(sim, 0.5, res.mean_modularity, n_null=100, seed=3)
    assert null.p_value < 0.05 and null.z > 2


def test_assignment_confidence_in_unit_interval():
    res = s.consensus_partition(_karate_sim(), threshold=0.5, runs=50, seed=4)
    conf = s.assignment_confidence(res)
    assert conf.shape == (34,) and np.all((conf >= 0) & (conf <= 1))
