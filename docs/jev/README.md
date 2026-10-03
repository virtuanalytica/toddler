# Own Jev server

`jevserver/` is a TypeSafe System One-compatible server (`POST /v1/systemone`), so `toddler/jev.py` and the virtualpc-jev-finance client work against it by changing only the endpoint and key.

## Run

```bash
export PYTHONPATH=.
python3 -m jevserver keygen --label toddler        # prints the key once; only its sha256 is stored
python3 -m jevserver serve --port 8092 --calibration docs/jev/piqa_calibration.json
```

The caller sets the printed key as `TYPESAFE_API_KEY` and uses `http://127.0.0.1:8092/v1/systemone` as endpoint.
Keys can be listed and revoked with `python3 -m jevserver list` / `revoke <id>`.

## Model and resources

The backend reads token probabilities from a llama.cpp server (`TODDLER_JEV_LLAMA_URL`, default the shared `llama-server` on `127.0.0.1:8011`, started with `--fit on` so it only uses free GPU memory).
No second model is started for Jev.

## Calibration (measured 2026-10-03, PIQA validation, 150 items, 300 yes/no questions)

| | Brier | Log loss | ECE | PIQA pair accuracy |
|---|---|---|---|---|
| Raw model | 0.257 | 1.165 | 0.252 | 0.907 |
| Isotonic calibration | 0.199 | 0.726 | 0.066 | 0.867 |

Raw probabilities rank the right solution well but are overconfident; the isotonic layer fixes the calibration and costs some pair accuracy because it creates ties.
Reproduce with `python3 -m jevserver.calibrate --items 150 --out docs/jev/piqa_calibration.json`.

## Latency

Measured end to end through `toddler/jev.py` for the four default reflex questions: 1385 ms on a cold request, 5 ms from the server cache.
Only cached answers meet the 20 ms reflex budget; a cold request makes the reflex stop, which is the intended fail-safe.
