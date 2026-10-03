"""Run the pre-registered behaviour-cloning peer experiment and store the report.

Run: PYTHONPATH=. python3 scripts/peer_bc_experiment.py  (writes docs/learn/peer_bc.json)
"""

import json
import time
from pathlib import Path

import torch

from toddler import business, resources
from toddler.learn import peer


def main() -> None:
    host = resources.probe()
    threads = resources.cpu_threads(host.cores, host.load_1m)
    torch.set_num_threads(min(threads, 8))
    t0 = time.time()
    result = peer.run_bc()
    report = {**result.to_dict(), "seconds": round(time.time() - t0, 1), "device": "cpu",
              "torch_threads": torch.get_num_threads(),
              **business.FIELDS}
    out = Path(__file__).resolve().parents[1] / "docs" / "learn" / "peer_bc.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
