"""Cron-safe, no-stake daily update and prospective prediction cycle."""

from __future__ import annotations

import argparse
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path

from toddler.finance_agents import (DEFAULT_DATA, DEFAULT_OFFICIAL, DEFAULT_OUT,
                                    atomic_json, leaderboard_status, predict_live,
                                    refresh_status, sync_official)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--official-destination', type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--account', default='develuse')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.finance-daily-cycle.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('finance daily cycle already running')
        cycle = {'schema': 'toddler-finance-daily-cycle/v1',
                 'started_utc': datetime.now(timezone.utc).isoformat(),
                 'existing_numerai_refresh': refresh_status(args.data_root),
                 'competitions': {}}
        failed = False
        for competition in ('crypto', 'signals'):
            steps = {}
            for name, operation in (
                ('official_sync', lambda: sync_official(competition, args.official_destination)),
                ('local_prediction', lambda: predict_live(competition, args.out, args.official_destination)),
                ('account_leaderboard', lambda: leaderboard_status(competition, args.out, args.account)),
            ):
                try:
                    steps[name] = {'status': 'ok', 'result': operation()}
                except Exception as exc:
                    failed = True
                    steps[name] = {'status': 'error', 'type': type(exc).__name__, 'message': str(exc)}
                    if name == 'official_sync':
                        break  # Never score an older live file after a failed sync.
            cycle['competitions'][competition] = steps
        cycle['finished_utc'] = datetime.now(timezone.utc).isoformat()
        cycle['status'] = 'failed' if failed else 'complete'
        stamp = cycle['started_utc'].replace('-', '').replace(':', '').replace('+00:00', 'Z').split('.')[0]
        report = args.out / 'daily_cycles' / f'{stamp}.json'
        atomic_json(report, cycle)
        print(json.dumps({'status': cycle['status'], 'report': str(report)}))
        if failed:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
