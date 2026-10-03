"""Reproduce every measured figure the whitepaper quotes from this repository's code.

Run: PYTHONPATH=. python3 scripts/measure_whitepaper_figures.py  (writes docs/whitepaper/figures.json)
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.datasets import load_breast_cancer

from toddler import evaluation as ev
from toddler import structure as s


def main() -> None:
    g = nx.karate_club_graph()
    adj = nx.to_numpy_array(g, weight=None)
    np.fill_diagonal(adj, 1)
    res = s.consensus_partition(adj, 0.5, runs=1000, seed=1)
    truth = np.array([g.nodes[i]["club"] == "Mr. Hi" for i in range(34)])
    purity = sum(max(np.sum(truth[res.labels == k]), np.sum(~truth[res.labels == k]))
                 for k in np.unique(res.labels)) / 34
    null = s.modularity_null_test(adj, 0.5, res.mean_modularity, n_null=1000, seed=3)
    conf = s.assignment_confidence(res)
    d = load_breast_cancer()
    sel = ev.nested_feature_selection(d.data[:, :10], d.target, outer=10, inner=10, seed=0)
    out = {
        "karate": {"groups": int(len(np.unique(res.labels))), "purity": round(float(purity), 3),
                   "mean_modularity": round(res.mean_modularity, 3), "consensus_runs": 1000, "null_graphs": 1000,
                   "null": "degree-preserving (double edge swap, weights permuted)",
                   "null_mean": round(null.null_mean, 3), "z": round(null.z, 1), "p": round(null.p_value, 4),
                   "share_confidence_gt_0_75": round(float(np.mean(conf > 0.75)), 2)},
        "breast_cancer_nested_cv": {"accuracy": round(sel.accuracy, 3), "sensitivity": round(sel.sensitivity, 3),
                                    "specificity": round(sel.specificity, 3),
                                    "selected": [str(d.feature_names[j]) for j in sel.selected]},
    }
    path = Path(__file__).resolve().parents[1] / "docs" / "whitepaper" / "figures.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
