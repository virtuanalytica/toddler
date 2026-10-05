"""Contamination-free evaluation: secret seed sets with commit-reveal.

The public EVAL_SEEDS are fixed and published, so a training procedure could (by accident or tuning)
end up fitted to them. A secret seed set is drawn from a band no training run ever samples
(tasks.SECRET_SEED_LOW..SECRET_SEED_HIGH), kept outside the repository, and only its commitment
is published:

    commitment = sha256("toddler-seeds-v1" | salt | comma-joined sorted seeds)

After the evaluation window closes the seeds and salt are revealed; anyone can then recompute the
commitment and re-run the evaluation. Until then nobody (including the code that trains toddlers)
can know which levels will be scored. The private file lives under
$TODDLER_SECRET_SEEDS (default ~/.local/share/toddler/secret_seeds), mode 0600.

    python3 -m toddler.learn.secret_seeds new --n 30 --window 2026-10-12
    python3 -m toddler.learn.secret_seeds reveal <set_id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from toddler.learn import tasks as T

DOMAIN = "toddler-seeds-v1"


@dataclass(frozen=True)
class Commitment:
    """The public half: safe to publish on the dashboard and in reports."""
    set_id: str
    n: int
    seed_low: int
    seed_high: int
    created_at: str
    reveal_after: str          # ISO date; seeds stay secret until the window has closed
    commitment: str


def commit(seeds: list[int], salt: str) -> str:
    body = f"{DOMAIN}|{salt}|{','.join(str(s) for s in sorted(seeds))}"
    return hashlib.sha256(body.encode()).hexdigest()


def verify(c: Commitment, seeds: list[int], salt: str) -> bool:
    in_band = all(c.seed_low <= s < c.seed_high for s in seeds)
    return in_band and len(set(seeds)) == c.n == len(seeds) and commit(seeds, salt) == c.commitment


def store_dir() -> Path:
    return Path(os.environ.get("TODDLER_SECRET_SEEDS")
                or Path.home() / ".local" / "share" / "toddler" / "secret_seeds")


def new_set(n: int, reveal_after: date, directory: Path | None = None) -> Commitment:
    """Draw n distinct seeds with the OS CSPRNG, store them privately, return the public commitment."""
    if n < 1:
        raise ValueError("n must be positive")
    lo, hi = T.SECRET_SEED_LOW, T.SECRET_SEED_HIGH
    seeds: set[int] = set()
    while len(seeds) < n:
        seeds.add(lo + secrets.randbelow(hi - lo))
    salt = secrets.token_hex(16)
    now = datetime.now(timezone.utc)
    set_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
    c = Commitment(set_id, n, lo, hi, now.isoformat(timespec="seconds"), reveal_after.isoformat(),
                   commit(sorted(seeds), salt))
    d = Path(directory) if directory else store_dir()
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = d / f"{set_id}.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)   # never overwrite, never world-readable
    with os.fdopen(fd, "w") as fh:
        json.dump({"commitment": asdict(c), "seeds": sorted(seeds), "salt": salt}, fh)
    return c


def load_private(set_id: str, directory: Path | None = None) -> tuple[Commitment, tuple[int, ...], str]:
    """For the evaluator only. The returned seeds must not be logged or published before reveal_after."""
    d = Path(directory) if directory else store_dir()
    blob = json.loads((d / f"{set_id}.json").read_text())
    c = Commitment(**blob["commitment"])
    seeds, salt = tuple(blob["seeds"]), blob["salt"]
    if not verify(c, list(seeds), salt):
        raise ValueError(f"secret seed set {set_id} does not match its own commitment (tampered?)")
    return c, seeds, salt


def reveal(set_id: str, today: date | None = None, directory: Path | None = None) -> dict:
    """Publishable reveal record; refuses before the window has closed."""
    c, seeds, salt = load_private(set_id, directory)
    today = today or datetime.now(timezone.utc).date()
    if today <= date.fromisoformat(c.reveal_after):
        raise PermissionError(f"set {set_id} may be revealed after {c.reveal_after}, not on {today}")
    return {"commitment": asdict(c), "seeds": list(seeds), "salt": salt}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m toddler.learn.secret_seeds")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("new"); a.add_argument("--n", type=int, default=30); a.add_argument("--window", required=True)
    b = sub.add_parser("reveal"); b.add_argument("set_id")
    args = ap.parse_args(argv)
    if args.cmd == "new":
        print(json.dumps(asdict(new_set(args.n, date.fromisoformat(args.window))), indent=2))
    else:
        print(json.dumps(reveal(args.set_id), indent=2))


if __name__ == "__main__":
    main()
