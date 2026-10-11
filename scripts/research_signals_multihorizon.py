"""Research-only Signals v3.0 model mixture over Jupiter 20/60 targets."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from toddler.finance_agents import (DEFAULT_OFFICIAL, assign_splits, atomic_json,
                                    engineered_features, per_date_corr, sha256)


OUT = Path('/media/knight2/claude-data/knight1/finance-agent-runs/signals_multihorizon_v1')
TARGETS = ('target_jupiter_60', 'target_jupiter_20')


def blend_ranked(dates, primary, auxiliary, primary_weight: float = 0.5) -> np.ndarray:
    """Blend within-date ranks; a global rank would mix different market weeks."""
    if not 0 <= primary_weight <= 1:
        raise ValueError('blend weight must be between zero and one')
    frame = pd.DataFrame({'date': pd.to_datetime(dates),
                          'primary': np.asarray(primary), 'auxiliary': np.asarray(auxiliary)})
    first = frame.groupby('date')['primary'].rank(pct=True, method='average')
    second = frame.groupby('date')['auxiliary'].rank(pct=True, method='average')
    return (primary_weight * first + (1 - primary_weight) * second).to_numpy()


def date_scores(frame: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
    values = []
    scored = pd.DataFrame({'date': frame.date.to_numpy(),
                           'target': frame.target_jupiter_60.to_numpy(),
                           'prediction': predictions})
    for _, group in scored.groupby('date', sort=True):
        if len(group) >= 20 and group.target.nunique() > 1:
            values.append(group.target.corr(group.prediction, method='spearman'))
    return np.asarray(values, dtype=float)


def paired_interval(first: np.ndarray, second: np.ndarray) -> dict:
    if len(first) != len(second) or len(first) < 20:
        raise ValueError('paired Signals date scores missing')
    difference = second - first
    rng = np.random.default_rng(20261011)
    length = 9  # 60-day target / approximately seven days between Signals eras
    starts = rng.integers(0, len(difference) - length + 1,
                          size=(2000, math.ceil(len(difference) / length)))
    index = (starts[:, :, None] + np.arange(length)).reshape(2000, -1)[:, :len(difference)]
    means = difference[index].mean(axis=1)
    return {'mean_delta': float(difference.mean()), 'date_count': len(difference),
            'block_bootstrap_95pct_ci': [float(np.quantile(means, .025)), float(np.quantile(means, .975))]}


def read_development(official: Path, features: list[str]) -> pd.DataFrame:
    """Materialize only dates through 2024, leaving the opened 2025+ data unused."""
    columns = ['date', *TARGETS, *features]
    frames = []
    for name in ('train.parquet', 'validation.parquet'):
        path = official / 'signals/v3.0' / name
        table = ds.dataset(str(path), format='parquet').to_table(
            columns=columns, filter=ds.field('date') < '2025-01-01')
        frames.append(table.to_pandas())
    frame = pd.concat(frames, ignore_index=True)
    frame['date'] = pd.to_datetime(frame['date'])
    frame = frame.loc[frame.target_jupiter_60.notna() & frame.target_jupiter_20.notna()].copy()
    frame['_part'] = assign_splits(frame.date, 'signals')
    return frame


def train(official: Path, out: Path, threads: int) -> dict:
    import joblib
    import lightgbm as lgb

    train_file = official / 'signals/v3.0/train.parquet'
    validation_file = official / 'signals/v3.0/validation.parquet'
    hashes = {'train': sha256(train_file), 'validation': sha256(validation_file)}
    features = [x for x in pq.ParquetFile(train_file).schema_arrow.names
                if x.startswith('feature_') and x != 'feature_country']
    frame = read_development(official, features)
    train_df = frame.loc[frame._part == 'train'].copy()
    dev_df = frame.loc[frame._part == 'dev'].copy()
    if len(train_df) < 5_000_000 or len(dev_df) < 500_000:
        raise ValueError('unexpectedly few Signals rows')
    train_df, derived = engineered_features(train_df, 'signals')
    dev_df, _ = engineered_features(dev_df, 'signals')
    features.extend(derived)
    for subset in (train_df, dev_df):
        for feature in features:
            subset[feature] = pd.to_numeric(subset[feature], errors='coerce').astype('float32')
    out.mkdir(parents=True, exist_ok=True)
    models = {}
    predictions = {}
    for target in TARGETS:
        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.035, num_leaves=20,
                                  min_child_samples=200, colsample_bytree=0.8, reg_lambda=10.0,
                                  n_jobs=threads, verbosity=-1, deterministic=True,
                                  force_col_wise=True, random_state=20261011)
        model.fit(train_df[features], train_df[target].astype('float32'))
        predictions[target] = model.predict(dev_df[features])
        path = out / f'{target}.joblib'
        joblib.dump(model, path)
        models[target] = {'path': str(path), 'sha256': sha256(path)}
    if sha256(train_file) != hashes['train'] or sha256(validation_file) != hashes['validation']:
        raise ValueError('Signals source changed during training')
    base = predictions['target_jupiter_60']
    auxiliary = predictions['target_jupiter_20']
    mixture = blend_ranked(dev_df.date, base, auxiliary)
    rng = np.random.default_rng(20261011)
    shuffled = auxiliary.copy()
    for _, indexes in dev_df.groupby('date').indices.items():
        shuffled[indexes] = rng.permutation(shuffled[indexes])
    shuffled_mix = blend_ranked(dev_df.date, base, shuffled)
    score = {name: per_date_corr(dev_df, values, 'target_jupiter_60', 'date', 60)
             for name, values in [('primary_60', base), ('auxiliary_20', auxiliary),
                                  ('rank_blend_50_50', mixture), ('shuffled_auxiliary_control', shuffled_mix)]}
    result = {'schema': 'toddler-signals-multihorizon/v1',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'train_sha256': hashes['train'], 'validation_sha256': hashes['validation'],
              'target_primary': 'target_jupiter_60', 'target_auxiliary': 'target_jupiter_20',
              'features': features, 'models': models, 'train_rows': len(train_df),
              'dev_rows': len(dev_df), 'score': score,
              'paired_blend_minus_primary': paired_interval(date_scores(dev_df, base),
                                                            date_scores(dev_df, mixture)),
              'claim': '2023-2024 development only; 2025+ already opened for prior baseline, so no new holdout claim'}
    atomic_json(out / 'experiment.json', result)
    return result


def predict(official: Path, out: Path) -> dict:
    import joblib

    result = json.loads((out / 'experiment.json').read_text())
    live = official / 'signals/v3.0/live.parquet'
    source_hash = sha256(live)
    frame = pq.read_table(live, columns=['date', 'numerai_ticker', *[c for c in result['features'] if c.startswith('feature_')]]).to_pandas(ignore_metadata=True)
    frame['date'] = pd.to_datetime(frame['date'])
    if frame.numerai_ticker.isna().any() or frame.numerai_ticker.duplicated().any():
        raise ValueError('missing or repeated Signals ticker')
    frame, _ = engineered_features(frame, 'signals')
    predictions = {}
    for target in TARGETS:
        entry = result['models'][target]
        if sha256(Path(entry['path'])) != entry['sha256']:
            raise ValueError(f'model changed: {target}')
        predictions[target] = joblib.load(entry['path']).predict(frame[result['features']])
    mixture = blend_ranked(frame.date, predictions['target_jupiter_60'], predictions['target_jupiter_20'])
    if not np.isfinite(mixture).all():
        raise ValueError('invalid Signals mixture prediction')
    dest = out / f'prospective-{source_hash[:12]}-{result["models"]["target_jupiter_60"]["sha256"][:12]}-{result["models"]["target_jupiter_20"]["sha256"][:12]}.csv'
    if dest.exists():
        audit_path = dest.with_suffix('.json')
        if audit_path.exists():
            return json.loads(audit_path.read_text())
        raise FileExistsError(dest)
    with dest.open('x', encoding='utf-8') as stream:
        pd.DataFrame({'numerai_ticker': frame.numerai_ticker, 'prediction': mixture}).to_csv(stream, index=False)
    audit = {'schema': 'toddler-signals-multihorizon-prospective/v1',
             'created_utc': datetime.now(timezone.utc).isoformat(),
             'source_date': str(frame.date.max().date()), 'rows': len(frame),
             'source_sha256': source_hash, 'prediction_sha256': sha256(dest),
             'prediction_path': str(dest), 'models': result['models'],
             'submitted': False, 'stake': 0,
             'status': 'local_research_forecast_waiting_for_60_day_target'}
    atomic_json(dest.with_suffix('.json'), audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'predict'))
    parser.add_argument('--official', type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    answer = train(args.official, args.out, args.threads) if args.action == 'train' else predict(args.official, args.out)
    print(json.dumps(answer, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
