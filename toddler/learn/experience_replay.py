"""Bounded, provenance-checked replay from a candidate's own training episodes.

This is public candidate research. Private evaluation seeds and official
generation weights never enter the buffer; a caller must copy the policy
before self-imitation. PPO remains on-policy and does not consume this buffer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from toddler.learn import tasks as T


@dataclass(frozen=True)
class Trajectory:
    task: str
    seed: int
    model_sha256: str
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    episode_return: float
    collected_utc: str

    def __post_init__(self) -> None:
        if self.task not in T.TASKS or type(self.seed) is not int or self.seed < T.TRAIN_SEED_LOW:
            raise ValueError("replay requires a registered task and a training-band seed")
        if len(self.model_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.model_sha256):
            raise ValueError("replay requires the source model's SHA-256")
        observations = np.asarray(self.observations, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.int64)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        if (observations.ndim != 2 or actions.ndim != 1 or len(actions) == 0
                or rewards.ndim != 1 or len(observations) != len(actions) or len(rewards) != len(actions)
                or not np.isfinite(observations).all() or not np.isfinite(rewards).all()
                or not np.isfinite(self.episode_return) or self.episode_return <= 0
                or abs(float(rewards.sum()) - self.episode_return) > 1e-5):
            raise ValueError("replay accepts only finite, successful complete episodes")
        try:
            collected = datetime.fromisoformat(self.collected_utc)
        except (TypeError, ValueError) as exc:
            raise ValueError("replay needs a valid collection timestamp") from exc
        if collected.tzinfo is None:
            raise ValueError("replay collection timestamp needs a timezone")
        for name, array in (("observations", observations), ("actions", actions), ("rewards", rewards)):
            immutable = array.copy()
            immutable.setflags(write=False)
            object.__setattr__(self, name, immutable)

    @property
    def sha256(self) -> str:
        header = json.dumps({"task": self.task, "seed": self.seed, "model_sha256": self.model_sha256,
                             "return": self.episode_return, "collected_utc": self.collected_utc,
                             "shape": self.observations.shape},
                            sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(header + self.observations.tobytes() + self.actions.tobytes()
                              + self.rewards.tobytes()).hexdigest()


class ReplayBuffer:
    def __init__(self, max_steps: int) -> None:
        if max_steps < 1:
            raise ValueError("replay step budget must be positive")
        self.max_steps = max_steps
        self.episodes: list[Trajectory] = []
        self.steps = 0

    def add(self, episode: Trajectory) -> None:
        if len(episode.actions) > self.max_steps:
            raise ValueError("one trajectory exceeds the replay budget")
        key = (episode.task, episode.seed, episode.model_sha256)
        if any((row.task, row.seed, row.model_sha256) == key for row in self.episodes):
            raise ValueError("duplicate replay episode")
        self.episodes.append(episode)
        self.steps += len(episode.actions)
        while self.steps > self.max_steps:
            removed = self.episodes.pop(0)
            self.steps -= len(removed.actions)

    def transitions(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        rows = [row for row in self.episodes if row.task == task]
        if not rows:
            raise ValueError(f"no successful replay for {task}")
        return np.concatenate([row.observations for row in rows]), np.concatenate([row.actions for row in rows])

    def manifest(self) -> dict:
        return {"schema": "toddler-own-training-replay/v1", "source": "own_training_interaction",
                "max_steps": self.max_steps, "retained_steps": self.steps,
                "episodes": [{"task": row.task, "seed": row.seed,
                              "model_sha256": row.model_sha256, "steps": len(row.actions),
                              "return": row.episode_return, "collected_utc": row.collected_utc,
                              "sha256": row.sha256}
                             for row in self.episodes]}

    def save(self, out: Path) -> dict:
        """Store the exact data outside Git; manifest binds every row to its source."""
        if not self.episodes:
            raise ValueError("cannot archive an empty replay buffer")
        out.mkdir(mode=0o700, parents=True, exist_ok=False)
        payload = {}
        for index, row in enumerate(self.episodes):
            payload[f"obs_{index}"] = row.observations
            payload[f"actions_{index}"] = row.actions
            payload[f"rewards_{index}"] = row.rewards
        archive = out / "trajectories.npz"
        np.savez_compressed(archive, **payload)
        result = {**self.manifest(), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
        (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    @classmethod
    def load(cls, out: Path) -> "ReplayBuffer":
        """Fail closed if the archive, row hashes or seed provenance changed."""
        metadata = json.loads((out / "manifest.json").read_text())
        archive = out / "trajectories.npz"
        if (metadata.get("schema") != "toddler-own-training-replay/v1"
                or metadata.get("source") != "own_training_interaction"
                or hashlib.sha256(archive.read_bytes()).hexdigest() != metadata.get("archive_sha256")):
            raise ValueError("replay archive or source provenance differs")
        buffer = cls(metadata["max_steps"])
        rows = metadata["episodes"]
        with np.load(archive, allow_pickle=False) as stored:
            expected = {f"{kind}_{index}" for index in range(len(rows))
                        for kind in ("obs", "actions", "rewards")}
            if set(stored.files) != expected:
                raise ValueError("replay archive has missing or unexpected arrays")
            for index, row in enumerate(rows):
                episode = Trajectory(row["task"], row["seed"], row["model_sha256"],
                                     stored[f"obs_{index}"], stored[f"actions_{index}"],
                                     stored[f"rewards_{index}"], row["return"], row["collected_utc"])
                if episode.sha256 != row["sha256"] or len(episode.actions) != row["steps"]:
                    raise ValueError("replay row differs from its manifest")
                buffer.add(episode)
        if buffer.steps != metadata["retained_steps"] or buffer.manifest()["episodes"] != rows:
            raise ValueError("replay row count or ordering changed")
        return buffer


def collect_successful(net, task: str, seeds: tuple[int, ...], model_sha256: str,
                       replay: ReplayBuffer) -> dict:
    """Sample candidate actions on training maps; keep only solved episodes."""
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < T.TRAIN_SEED_LOW for seed in seeds)):
        raise ValueError("replay collection requires unique training-band seeds")
    net.to("cpu").eval()
    solved = 0
    for seed in seeds:
        env = T.make(task)
        try:
            obs, _ = env.reset(seed=seed)
            generator = torch.Generator().manual_seed(seed)
            observations, actions, rewards = [], [], []
            total = 0.0
            done = False
            while not done:
                with torch.no_grad():
                    logits = net(torch.as_tensor(obs, dtype=torch.float32))[0]
                    action = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=generator))
                observations.append(obs)
                actions.append(action)
                obs, reward, term, trunc, _ = env.step(action)
                rewards.append(reward)
                total += reward
                done = term or trunc
            if total > 0:
                replay.add(Trajectory(task, seed, model_sha256, np.asarray(observations),
                                      np.asarray(actions), np.asarray(rewards), float(total),
                                      datetime.now(timezone.utc).isoformat()))
                solved += 1
        finally:
            env.close()
    return {"attempted": len(seeds), "successful": solved,
            "retained_episodes": len(replay.episodes), "retained_steps": replay.steps}


def self_imitate(net, task: str, replay: ReplayBuffer, *, epochs: int = 4,
                 lr: float = 1e-4, minibatch: int = 256, seed: int = 0) -> dict:
    """Fit a copied specialist's task adapter/head to its own successful actions."""
    if epochs < 1 or lr <= 0 or minibatch < 1:
        raise ValueError("invalid self-imitation budget")
    features, labels = replay.transitions(task)
    if features.shape[1] != net.task_dims[task][0] or np.any(labels < 0) or np.any(labels >= net.task_dims[task][1]):
        raise ValueError("replay observations/actions do not match the candidate policy")
    trainable = [*net.adapters[task].parameters(), *net.pi[task].parameters()]
    prior_grad = [(parameter, parameter.requires_grad) for parameter in net.parameters()]
    try:
        for parameter, _ in prior_grad:
            parameter.requires_grad_(False)
        for parameter in trainable:
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(trainable, lr=lr)
        x = torch.as_tensor(features, dtype=torch.float32)
        y = torch.as_tensor(labels, dtype=torch.long)
        order = np.arange(len(y))
        rng = np.random.default_rng(seed)
        net.train()
        for _ in range(epochs):
            rng.shuffle(order)
            for offset in range(0, len(order), minibatch):
                batch = torch.as_tensor(order[offset:offset + minibatch])
                logits, _ = net.forward_task(task, x[batch])
                loss = torch.nn.functional.cross_entropy(logits, y[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        net.eval()
        with torch.no_grad():
            logits, _ = net.forward_task(task, x)
            accuracy = float((logits.argmax(-1) == y).float().mean())
    finally:
        for parameter, old in prior_grad:
            parameter.requires_grad_(old)
        net.eval()
    return {"method": "own_successful_trajectory_imitation", "epochs": epochs,
            "lr": lr, "minibatch": minibatch, "transitions": len(labels),
            "training_action_accuracy": accuracy,
            "replay_manifest_sha256": hashlib.sha256(json.dumps(replay.manifest(), sort_keys=True).encode()).hexdigest(),
            "uses_teacher_grid": False, "promotion_eligible": False}
