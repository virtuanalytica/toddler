"""Research-only Crypto v2 candidate with two-day-lagged YIEDL aggregates.

Historical YIEDL availability and revisions have not been independently
reconstructed. These results cannot promote a model without prospective data.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from toddler.finance_agents import (DEFAULT_OFFICIAL, assign_splits,
                                    atomic_json, engineered_features,
                                    per_date_corr, sha256)


YIEDL = Path('/media/knight2/EDS2/projects/numerai-signals/proofs/cross_pollination/yiedl_agg.parquet')
OUT = Path('/media/knight2/claude-data/knight1/finance-agent-runs/yiedl_crypto_v2')
YIEDL_COLS = ('yiedl_pvm_mean', 'yiedl_onchain_mean', 'yiedl_sent_mean')


def lagged_join(official: pl.DataFrame, yiedl: pl.DataFrame, days: int = 2) -> pl.DataFrame:
    """Use the most recent YIEDL row at least `days` before the Numerai date."""
    if days < 2:
        raise ValueError('YIEDL research protocol requires at least two days of source lag')
    left = (official.with_columns(pl.col('date').cast(pl.Date))
            .with_columns((pl.col('date') - pl.duration(days=days)).alias('cutoff'))
            .sort(['symbol', 'cutoff']))
    right = (yiedl.select(pl.col('date').cast(pl.Date).alias('source_date'),
                           'symbol', *YIEDL_COLS)
             .sort(['symbol', 'source_date']))
    joined = left.join_asof(right, left_on='cutoff', right_on='source_date',
                            by='symbol', strategy='backward', tolerance='7d')
    violations = joined.filter(pl.col('source_date') > pl.col('cutoff')).height
    if violations:
        raise ValueError(f'{violations} future YIEDL observations entered the join')
    return joined


def load_yiedl(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path, columns=['date', 'symbol', *YIEDL_COLS])


def _date_scores(frame: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
    values = []
    scored = pd.DataFrame({'date': frame['date'].to_numpy(), 'target': frame['target_binned_return_20'].to_numpy(),
                           'prediction': prediction})
    for _, group in scored.groupby('date', sort=True):
        if len(group) >= 20 and group.target.nunique() > 1:
            values.append(group.target.corr(group.prediction, method='spearman'))
    return np.array(values, dtype=float)


def _paired_interval(first: np.ndarray, second: np.ndarray) -> dict:
    if len(first) != len(second) or len(first) < 20:
        raise ValueError('paired date scores missing')
    delta = second - first
    rng = np.random.default_rng(20261011)
    length = 20  # overlapping Crypto 20-day targets
    starts = rng.integers(0, len(delta) - length + 1, size=(2000, math.ceil(len(delta) / length)))
    samples = (starts[:, :, None] + np.arange(length)).reshape(2000, -1)[:, :len(delta)]
    means = delta[samples].mean(axis=1)
    return {'mean_delta': float(delta.mean()), 'block_bootstrap_95pct_ci':
            [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            'dates': int(len(delta))}


def train(official: Path, yiedl: Path, out: Path, threads: int) -> dict:
    """Train on pre-2025, select on 2025 only; do not load 2026 labels."""
    import joblib
    import lightgbm as lgb

    official_file = official / 'crypto/v2.0/train.parquet'
    official_hash = sha256(official_file)
    yiedl_hash = sha256(yiedl)
    schema = pl.read_parquet_schema(official_file)
    base_cols = [c for c in schema if c.startswith('feature_')]
    cols = ['date', 'symbol', 'target_binned_return_20', *base_cols]
    # Pushdown precedes collect: no 2026 holdout targets are read.
    old = (pl.scan_parquet(official_file).select(cols)
           .filter(pl.col('date') <= pl.datetime(2025, 12, 31))
           .collect())
    joined = lagged_join(old, load_yiedl(yiedl))
    frame = joined.to_pandas()
    frame['date'] = pd.to_datetime(frame['date'])
    frame = frame.loc[frame.target_binned_return_20.notna()].copy()
    frame['_part'] = assign_splits(frame['date'], 'crypto')
    train_df = frame.loc[frame._part == 'train'].copy()
    dev_df = frame.loc[frame._part == 'dev'].copy()
    if len(train_df) < 300_000 or len(dev_df) < 50_000:
        raise ValueError('unexpectedly few train/development rows')
    train_df, derived = engineered_features(train_df, 'crypto')
    dev_df, _ = engineered_features(dev_df, 'crypto')
    base_features = base_cols + derived
    research_features = base_features + list(YIEDL_COLS)
    for part in (train_df, dev_df):
        for feature in research_features:
            part[feature] = pd.to_numeric(part[feature], errors='coerce').astype('float32')
    out.mkdir(parents=True, exist_ok=True)
    models = {}
    results = {}
    predictions = {}
    for name, features in [('official_only', base_features), ('official_plus_yiedl', research_features)]:
        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.035, num_leaves=20,
                                  min_child_samples=200, colsample_bytree=0.8, reg_lambda=10,
                                  n_jobs=threads, verbosity=-1, deterministic=True,
                                  force_col_wise=True, random_state=20261011)
        model.fit(train_df[features], train_df['target_binned_return_20'])
        pred = model.predict(dev_df[features])
        predictions[name] = pred
        results[name] = per_date_corr(dev_df, pred, 'target_binned_return_20', 'date', 20)
        path = out / f'{name}.joblib'
        joblib.dump(model, path)
        models[name] = {'path': str(path), 'sha256': sha256(path), 'features': features}
    delta = _paired_interval(_date_scores(dev_df, predictions['official_only']),
                             _date_scores(dev_df, predictions['official_plus_yiedl']))
    if sha256(official_file) != official_hash or sha256(yiedl) != yiedl_hash:
        raise ValueError('source changed during research training')
    shuffled = dev_df[research_features].copy()
    rng = np.random.default_rng(20261011)
    for _, indexes in dev_df.groupby('date').indices.items():
        for col in YIEDL_COLS:
            shuffled.iloc[indexes, shuffled.columns.get_loc(col)] = rng.permutation(shuffled.iloc[indexes][col].to_numpy())
    control_prediction = joblib.load(models['official_plus_yiedl']['path']).predict(shuffled)
    result = {'schema': 'toddler-yiedl-crypto-research/v1',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'official_sha256': official_hash, 'yiedl_sha256': yiedl_hash,
              'source_lag_days': 2, 'source_max_age_days': 7,
              'point_in_time_status': 'historical YIEDL backfill/revision audit incomplete',
              'train_rows': len(train_df), 'dev_rows': len(dev_df),
              'dev_yiedl_coverage': {c: float(dev_df[c].notna().mean()) for c in YIEDL_COLS},
              'models': models, 'development_scores': results,
              'paired_yiedl_minus_official': delta,
              'within_date_shuffled_yiedl_control': per_date_corr(
                  dev_df, control_prediction, 'target_binned_return_20', 'date', 20),
              'claim': 'exploratory development comparison only; 2026 holdout was already opened for a prior candidate'}
    atomic_json(out / 'experiment.json', result)
    return result


def predict(official: Path, yiedl: Path, out: Path) -> dict:
    """Write a local research forecast with actual YIEDL source dates and coverage."""
    import joblib

    experiment = json.loads((out / 'experiment.json').read_text())
    choice = experiment['models']['official_plus_yiedl']
    model_path = Path(choice['path'])
    if sha256(model_path) != choice['sha256']:
        raise ValueError('research model changed')
    live = official / 'crypto/v2.0/live.parquet'
    source_hash = sha256(live)
    yiedl_hash = sha256(yiedl)
    cols = ['date', 'symbol', *[c for c in choice['features'] if c.startswith('feature_')]]
    joined = lagged_join(pl.read_parquet(live, columns=cols), load_yiedl(yiedl))
    frame = joined.to_pandas()
    if frame['yiedl_pvm_mean'].notna().mean() < 0.8 or frame['yiedl_sent_mean'].notna().mean() < 0.8:
        raise ValueError('YIEDL live coverage below 80% research gate')
    frame, _ = engineered_features(frame, 'crypto')
    if frame.symbol.isna().any() or frame.symbol.duplicated().any():
        raise ValueError('missing or repeated symbol')
    pred = joblib.load(model_path).predict(frame[choice['features']])
    if not np.isfinite(pred).all():
        raise ValueError('invalid research prediction')
    ranks = pd.Series(pred).rank(pct=True, method='average')
    if sha256(live) != source_hash or sha256(yiedl) != yiedl_hash:
        raise ValueError('source changed during prospective prediction')
    dest = out / f'prospective-{source_hash[:12]}-{yiedl_hash[:12]}-{choice["sha256"][:12]}.csv'
    if dest.exists():
        audit_path = dest.with_suffix('.json')
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if audit.get('official_live_sha256') == source_hash and audit.get('yiedl_sha256') == yiedl_hash:
                return audit
        raise FileExistsError(dest)
    with dest.open('x', encoding='utf-8') as stream:
        pd.DataFrame({'symbol': frame.symbol.to_numpy(), 'prediction': ranks.to_numpy()}).to_csv(stream, index=False)
    audit = {'schema': 'toddler-yiedl-crypto-prospective/v1',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'rows': len(frame),
             'source_date': str(pd.to_datetime(frame.date).max().date()),
             'yiedl_coverage': {c: float(frame[c].notna().mean()) for c in YIEDL_COLS},
             'latest_used_yiedl_date': str(pd.to_datetime(frame.source_date).max().date()),
             'official_live_sha256': source_hash, 'yiedl_sha256': yiedl_hash,
             'model_sha256': choice['sha256'], 'prediction_sha256': sha256(dest),
             'prediction_path': str(dest), 'submitted': False, 'stake': 0,
             'point_in_time_status': experiment['point_in_time_status']}
    atomic_json(dest.with_suffix('.json'), audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'predict'))
    parser.add_argument('--official', type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument('--yiedl', type=Path, default=YIEDL)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    result = train(args.official, args.yiedl, args.out, args.threads) if args.action == 'train' else predict(args.official, args.yiedl, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
