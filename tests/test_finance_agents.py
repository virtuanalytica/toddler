from __future__ import annotations

import pandas as pd
import pytest

from toddler.finance_agents import assign_splits, engineered_features, per_date_corr, prepare


def test_crypto_split_has_embargoes():
    dates = pd.Series(pd.to_datetime([
        '2024-12-31', '2025-01-15', '2025-03-01',
        '2025-12-31', '2026-01-15', '2026-03-01',
    ]))
    assert assign_splits(dates, 'crypto').tolist() == [
        'train', 'embargo', 'dev', 'dev', 'embargo', 'holdout',
    ]


def test_feature_engineering_only_uses_same_row():
    frame = pd.DataFrame({
        'feature_momentum_20d': [0.8, 0.2],
        'feature_momentum_60d': [0.4, 0.3],
        'target_binned_return_20': [0.0, 1.0],
    })
    made, names = engineered_features(frame, 'crypto')
    assert names == ['derived_momentum_20_minus_60']
    assert made[names[0]].tolist() == pytest.approx([0.4, -0.1])


def test_prepare_excludes_targets_and_embargo(tmp_path):
    root = tmp_path / 'data'
    path = root / 'live_current' / 'crypto_v20_train.parquet'
    path.parent.mkdir(parents=True)
    pd.DataFrame({
        'date': pd.to_datetime(['2024-12-01', '2025-01-01', '2025-05-01', '2026-04-01']),
        'feature_momentum_20d': [1.0, 2.0, 3.0, 4.0],
        'target_binned_return_20': [0.5, 0.5, 0.75, 1.0],
        'target_binned_return_60': [0.5, 0.5, 0.75, 1.0],
    }).to_parquet(path, index=False)
    manifest = prepare('crypto', root, tmp_path / 'out')
    assert [manifest['partitions'][x]['rows'] for x in ('train', 'dev', 'holdout')] == [1, 1, 1]
    assert manifest['features'] == ['feature_momentum_20d']
    assert 'target_binned_return_60' not in manifest['features']
    assert pd.read_parquet(manifest['partitions']['holdout']['path'])['date'].iloc[0] == pd.Timestamp('2026-04-01')


def test_signal_split_respects_60_day_horizon():
    dates = pd.Series(pd.to_datetime(['2022-12-30', '2023-02-01', '2023-04-07',
                                      '2024-12-27', '2025-02-01', '2025-04-04']))
    assert assign_splits(dates, 'signals').tolist() == [
        'train', 'embargo', 'dev', 'dev', 'embargo', 'holdout',
    ]


def test_correlated_prediction_has_block_interval():
    frame = pd.DataFrame({
        'date': [pd.Timestamp('2025-01-01') + pd.Timedelta(days=i)
                 for i in range(15) for _ in range(30)],
        'target': list(range(30)) * 15,
    })
    result = per_date_corr(frame, frame['target'].to_numpy(), 'target', 'date', 5)
    assert result['mean_date_spearman'] == pytest.approx(1.0)
    assert result['block_bootstrap_95pct_ci'] == pytest.approx([1.0, 1.0])
    assert result['bootstrap_block_dates'] == 5
