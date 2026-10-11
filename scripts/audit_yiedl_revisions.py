"""Capture prospective YIEDL feature snapshots and detect later revisions."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from toddler.finance_agents import atomic_json, sha256


DEFAULT_SOURCE = Path('/media/knight2/EDS2/projects/numerai-signals/proofs/cross_pollination/yiedl_agg.parquet')
DEFAULT_STATE = Path('/media/knight2/claude-data/knight1/finance-agent-runs/yiedl_source_audit')
FIELDS = ('yiedl_pvm_mean', 'yiedl_onchain_mean', 'yiedl_sent_mean')


def compare(current: pl.DataFrame, previous: pl.DataFrame,
            first: date, last: date) -> dict:
    """Compare the same date/symbol observations from two capture times."""
    keys = ['date', 'symbol']
    current = current.filter(pl.col('date').is_between(first, last))
    previous = previous.filter(pl.col('date').is_between(first, last))
    now = current.select(*keys, *[pl.col(c).alias(f'{c}_now') for c in FIELDS]).with_columns(pl.lit(True).alias('_now'))
    before = previous.select(*keys, *[pl.col(c).alias(f'{c}_before') for c in FIELDS]).with_columns(pl.lit(True).alias('_before'))
    joined = now.join(before, on=keys, how='full', coalesce=True)
    overlap = joined.filter(pl.col('_now').is_not_null() & pl.col('_before').is_not_null())
    changed = {}
    for field in FIELDS:
        changed[field] = int(overlap.select((~pl.col(f'{field}_now').eq_missing(pl.col(f'{field}_before'))).sum()).item())
    return {'overlap_rows': overlap.height,
            'new_keys_in_old_dates': joined.filter(pl.col('_before').is_null()).height,
            'removed_keys_in_old_dates': joined.filter(pl.col('_now').is_null()).height,
            'changed_values': changed,
            'revision_detected': any(changed.values()) or joined.filter(pl.col('_before').is_null() | pl.col('_now').is_null()).height > 0}


def audit(source: Path, state: Path, asof: date, lookback_days: int = 90) -> dict:
    if lookback_days < 30:
        raise ValueError('lookback must cover at least the longest near-term label window')
    state.mkdir(parents=True, exist_ok=True)
    with (state / '.audit.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('YIEDL revision audit already running') from exc
        source_hash = sha256(source)
        start = asof - timedelta(days=lookback_days)
        frame = (pl.scan_parquet(source)
                 .select('date', 'symbol', *FIELDS)
                 .filter(pl.col('date').is_between(start, asof))
                 .collect())
        if sha256(source) != source_hash:
            raise ValueError('YIEDL aggregate changed during snapshot')
        if frame.height == 0 or frame.select(pl.struct(['date', 'symbol']).n_unique()).item() != frame.height:
            raise ValueError('empty or duplicate-key YIEDL snapshot')
        latest = frame.select(pl.col('date').max()).item()
        if (asof - latest).days > 7:
            raise ValueError(f'YIEDL source stale by {(asof - latest).days} days')
        snapshot = state / 'snapshots' / f'{asof:%Y%m%d}-{source_hash[:12]}.parquet'
        report = snapshot.with_suffix('.json')
        if snapshot.exists() and report.exists():
            saved = json.loads(report.read_text())
            if saved.get('snapshot_sha256') == sha256(snapshot):
                return saved
            raise ValueError('existing YIEDL snapshot hash does not match its audit report')
        prior = sorted(path for path in (state / 'snapshots').glob('*.parquet') if path != snapshot) if snapshot.parent.exists() else []
        comparison = None
        previous_path = None
        if prior:
            previous_path = prior[-1]
            previous_asof = date.fromisoformat(previous_path.name[:4] + '-' + previous_path.name[4:6] + '-' + previous_path.name[6:8])
            common_start = max(start, previous_asof - timedelta(days=lookback_days))
            common_end = min(asof, previous_asof)
            comparison = compare(frame, pl.read_parquet(previous_path), common_start, common_end)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        temporary = snapshot.with_name(f'.{snapshot.name}.{os.getpid()}.tmp')
        frame.write_parquet(temporary)
        temporary.replace(snapshot)
        result = {'schema': 'toddler-yiedl-revision-audit/v1',
                  'captured_utc': datetime.now(timezone.utc).isoformat(),
                  'asof': asof.isoformat(), 'window_start': start.isoformat(),
                  'source_path': str(source), 'source_sha256': source_hash,
                  'snapshot_path': str(snapshot), 'snapshot_sha256': sha256(snapshot),
                  'rows': frame.height, 'latest_source_date': latest.isoformat(),
                  'previous_snapshot': str(previous_path) if previous_path else None,
                  'comparison': comparison,
                  'status': 'baseline_capture' if comparison is None else (
                      'revision_detected' if comparison['revision_detected'] else 'no_revision_in_overlap'),
                  'scope': 'prospective snapshot audit only; earlier historical revisions remain unknown'}
        atomic_json(report, result)
        atomic_json(state / 'latest.json', result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--state', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--asof', type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    parser.add_argument('--lookback-days', type=int, default=90)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.state, args.asof, args.lookback_days), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
