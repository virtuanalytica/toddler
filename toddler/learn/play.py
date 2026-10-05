"""Let a toddler from the generation register act, and measure it, in a form a chat can show.

Used by the mixture-of-models chat (virtualpc) to put a real toddler in front of the models:
`episode` plays one level and returns the trajectory as text (MiniGrid levels as an ASCII grid with
the toddler's arrow, classic-control tasks as their state vector), and `iq_probe` measures the
toddler's IQ quotient against the frozen G1 reference on exactly the fingerprint that reference was
frozen under (public EVAL_SEEDS, sampled actions, measured random anchors). Everything is computed,
nothing is narrated: the models can explain a result, they cannot change it.

    python3 -m toddler.learn.play episode G1/t1001 doorkey5 --seed 424242
    python3 -m toddler.learn.play iq G1/t1001
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from toddler import quotients
from toddler.learn import generations as G
from toddler.learn import tasks as T

REFERENCE = "G1"
REFERENCE_TASKS = ("cartpole", "acrobot", "empty5", "doorkey5")
REFERENCE_MODE = "sample"
GRID_ACTIONS = ("left", "right", "forward", "pickup", "drop", "toggle", "done")


def registry_root() -> Path:
    return Path(os.environ.get("TODDLER_GENERATIONS_ROOT") or "/media/knight2/EDS2/toddler-generations")


def load(ref: str, root: Path | None = None):
    gen, tid = ref.split("/", 1)
    return G.Registry(root or registry_root()).load(gen, tid)


def _policy(net, task: str):
    from toddler.learn.multitask import MultiTaskNet, TaskView

    if isinstance(net, MultiTaskNet):
        if task not in net.task_dims:
            raise KeyError(f"this toddler never learned {task!r}; it knows {sorted(net.task_dims)}")
        return TaskView(net, task)
    return net


def _frame(env, task: str, obs) -> str:
    if T.TASKS[task].grid:
        return env.unwrapped.pprint_grid()
    return "state " + ", ".join(f"{x:+.3f}" for x in np.asarray(obs, float).ravel())


def episode(ref: str, task: str, seed: int, mode: str = "greedy", frames: int = 6,
            root: Path | None = None) -> dict:
    """Play one level. `frames` evenly spaced snapshots (first and last always included)."""
    import torch

    net, rec = load(ref, root)
    pol = _policy(net, task).to("cpu").eval()
    env = T.make(task)
    obs, _ = env.reset(seed=seed)
    gen = torch.Generator().manual_seed(int(seed))
    snaps, actions, total, done = [_frame(env, task, obs)], [], 0.0, False
    while not done:
        with torch.no_grad():
            logits = pol(torch.as_tensor(obs, dtype=torch.float32))[0]
        a = int(torch.argmax(logits)) if mode == "greedy" else int(
            torch.multinomial(torch.softmax(logits, -1), 1, generator=gen))
        obs, r, term, trunc, _ = env.step(a)
        actions.append(GRID_ACTIONS[a] if T.TASKS[task].grid else a)
        snaps.append(_frame(env, task, obs))
        total += r
        done = term or trunc
    env.close()
    anchor = T.random_anchor(task, (seed,))
    keep = sorted({0, len(snaps) - 1, *np.linspace(0, len(snaps) - 1, max(frames, 2)).astype(int).tolist()})
    norm = None if T.TASKS[task].solved <= anchor else round(float(T.normalise(task, total, anchor)), 4)
    return {"toddler": ref, "weights_sha256": rec.weights_sha256, "task": task, "env_id": T.TASKS[task].env_id,
            "seed": int(seed), "public_eval_seed": int(seed) in T.EVAL_SEEDS, "mode": mode,
            "return": round(total, 4), "random_return_same_seed": round(anchor, 4), "normalised": norm,
            "solved": total >= T.TASKS[task].solved, "steps": len(actions), "actions": actions,
            "frames": [{"step": i, "view": snaps[i]} for i in keep]}


def iq_probe(ref: str, root: Path | None = None) -> dict:
    """IQ quotient of one toddler against the frozen G1 reference (100 = G1's typical toddler)."""
    from toddler.learn import scoring

    root = root or registry_root()
    net, rec = load(ref, root)
    anchors = {t: T.random_anchor(t) for t in REFERENCE_TASKS}
    fp = quotients.fingerprint(list(REFERENCE_TASKS), T.EVAL_SEEDS, [anchors[t] for t in REFERENCE_TASKS],
                               [T.TASKS[t].solved for t in REFERENCE_TASKS], eval_mode=REFERENCE_MODE)
    reference = G.Registry(root).reference(REFERENCE, fp)       # refuses a different fingerprint
    per_task = {}
    for t in REFERENCE_TASKS:
        raw = scoring.evaluate(_policy(net, t), t, mode=REFERENCE_MODE)
        per_task[t] = float(np.mean([T.normalise(t, x, anchors[t]) for x in raw]))
    raw = quotients.iq_raw(np.asarray([[per_task[t] for t in REFERENCE_TASKS]]))
    return {"toddler": ref, "weights_sha256": rec.weights_sha256, "reference": REFERENCE,
            "reference_fingerprint": fp.digest(), "tasks": list(REFERENCE_TASKS), "eval_seeds": len(T.EVAL_SEEDS),
            "per_task_normalised": {t: round(v, 4) for t, v in per_task.items()},
            "IQ_raw": round(raw, 4), "IQ_quotient": round(quotients.to_quotient(raw, reference), 1),
            "note": "single toddler, no confidence interval; 0 = random policy, 1 = solve threshold per task"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m toddler.learn.play")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("episode")
    e.add_argument("ref"); e.add_argument("task"); e.add_argument("--seed", type=int, required=True)
    e.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
    q = sub.add_parser("iq")
    q.add_argument("ref")
    a = ap.parse_args(argv)
    out = episode(a.ref, a.task, a.seed, a.mode) if a.cmd == "episode" else iq_probe(a.ref)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
