"""Generation register: every toddler that is trained is saved, traceable and comparable.

Layout: <root>/<generation>/<toddler_id>/
  weights.pt        state_dict of the policy
  meta.json         task, config, step budget, parents (lineage), normalised scores per
                    evaluation seed, hardware fingerprint, sha256 of the weights, business fields
  checkpoints/      intermediate weights every N updates (resume / audit)

Business fields (operator definition): VirtualV Holding B.V. invoices the generation;
virtuanalytica VOF supplies the compute and sells on commission.

Promotion of a candidate generation over the reference follows the replication culture of
toddler.evaluation: at least five seeds, one-sided Mann-Whitney p < alpha AND probability of
improvement >= 0.75.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from toddler import business, quotients
from toddler.learn import scoring
from toddler.learn.policy import ActorCritic

BUSINESS = business.FIELDS


@dataclass
class ToddlerRecord:
    generation: str
    toddler_id: str
    task: str
    config: dict
    steps: int
    parents: list[str]
    eval_scores: list[float]                      # normalised, per held-out seed
    hardware: dict
    device_switches: list[str] = field(default_factory=list)   # non-empty: not bitwise reproducible
    weights_sha256: str = ""
    created_at: float = field(default_factory=time.time)
    business: dict = field(default_factory=lambda: dict(BUSINESS))

    @property
    def score(self) -> float:
        return float(np.mean(self.eval_scores))


def hardware_fingerprint() -> dict:
    """Describe where the toddler was trained (quality never depends on it; efficiency does)."""
    import os
    import platform

    fp = {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(),
          "torch_threads": torch.get_num_threads(), "device": "cpu"}
    try:
        from toddler import resources

        fp["gpus"] = [g.name for g in resources.probe().gpus]
    except Exception as exc:  # fingerprint is descriptive; a probe failure is recorded, not hidden
        fp["gpus"] = f"probe failed: {type(exc).__name__}"
    return fp


class Registry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, generation: str, toddler_id: str) -> Path:
        return self.root / generation / toddler_id

    def save(self, net: ActorCritic, rec: ToddlerRecord) -> Path:
        d = self._dir(rec.generation, rec.toddler_id)
        d.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        torch.save({"spec": net.spec(), "state": net.state_dict()}, buf)
        data = buf.getvalue()
        rec.weights_sha256 = hashlib.sha256(data).hexdigest()
        (d / "weights.pt").write_bytes(data)
        (d / "meta.json").write_text(json.dumps(asdict(rec), indent=1))
        return d

    def checkpoint_fn(self, generation: str, toddler_id: str):
        cdir = self._dir(generation, toddler_id) / "checkpoints"

        def save(net: ActorCritic, log) -> None:
            cdir.mkdir(parents=True, exist_ok=True)
            torch.save({"spec": net.spec(), "state": net.state_dict(), "steps": log.steps},
                       cdir / f"step_{log.steps:08d}.pt")
        return save

    def load(self, generation: str, toddler_id: str) -> tuple[ActorCritic, ToddlerRecord]:
        d = self._dir(generation, toddler_id)
        data = (d / "weights.pt").read_bytes()
        rec = ToddlerRecord(**json.loads((d / "meta.json").read_text()))
        if hashlib.sha256(data).hexdigest() != rec.weights_sha256:
            raise ValueError(f"{generation}/{toddler_id}: weights do not match their recorded sha256")
        blob = torch.load(io.BytesIO(data), weights_only=True)
        net = ActorCritic(**blob["spec"])
        net.load_state_dict(blob["state"])
        return net, rec

    def freeze_reference(self, generation: str, fp: quotients.ReferenceFingerprint) -> Path:
        """Freeze a generation as the quotient reference: its per-toddler raw scores plus the
        fingerprint (tasks, eval seeds, anchors) they were measured on."""
        recs = self.generation(generation)
        if len(recs) < 2:
            raise ValueError("a reference generation needs at least two toddlers")
        path = self.root / generation / "reference.json"
        path.write_text(json.dumps({"generation": generation, "fingerprint": asdict(fp), "digest": fp.digest(),
                                    "raw": {r.toddler_id: r.score for r in recs}}, indent=1))
        return path

    def reference(self, generation: str, fp: quotients.ReferenceFingerprint) -> list[float]:
        """Per-toddler raw scores of the frozen reference; refuses a different fingerprint."""
        ref = json.loads((self.root / generation / "reference.json").read_text())
        frozen = quotients.ReferenceFingerprint(**{k: tuple(v) for k, v in ref["fingerprint"].items()})
        frozen.require_same(fp)
        return list(ref["raw"].values())

    def generation(self, generation: str) -> list[ToddlerRecord]:
        base = self.root / generation
        return [ToddlerRecord(**json.loads((d / "meta.json").read_text()))
                for d in sorted(base.iterdir()) if (d / "meta.json").exists()] if base.exists() else []

    def lineage(self, generation: str, toddler_id: str) -> list[str]:
        """Ancestors as 'generation/toddler_id', nearest first."""
        out, frontier = [], [f"{generation}/{toddler_id}"]
        while frontier:
            g, t = frontier.pop(0).split("/", 1)
            meta = json.loads((self._dir(g, t) / "meta.json").read_text())
            for p in meta["parents"]:
                if p not in out:
                    out.append(p)
                    frontier.append(p)
        return out


@dataclass(frozen=True)
class Promotion:
    promote: bool
    p_value: float
    prob_improvement: float
    candidate_iqm: float
    reference_iqm: float


def decide_promotion(candidate: list[float], reference: list[float], alpha: float = 0.05,
                     min_seeds: int = 5, min_prob: float = 0.75) -> Promotion:
    c, r = np.asarray(candidate, float), np.asarray(reference, float)
    if len(c) < min_seeds or len(r) < min_seeds:
        raise ValueError(f"promotion needs at least {min_seeds} seeds per generation")
    p = float(stats.mannwhitneyu(c, r, alternative="greater").pvalue)
    pi = scoring.prob_improvement(c[:, None], r[:, None])
    return Promotion(bool(p < alpha and pi >= min_prob), p, pi, scoring.iqm(c), scoring.iqm(r))
