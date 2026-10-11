"""Auditable four-role research pipeline for Numerai Crypto and Signals.

Only public, time-stamped features enter model training. This module does not
submit predictions, stake NMR, or promote a model from a historical backtest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_DATA = Path('/media/knight2/EDS2/projects/numerai-signals/data')
DEFAULT_OUT = Path('/media/knight2/claude-data/knight1/finance-agent-runs')
DEFAULT_OFFICIAL = Path('/media/knight2/claude-data/knight1/finance-agent-data/official')
SPECS = {
    'crypto': {
        'files': ['live_current/crypto_v20_train.parquet'],
        'date': 'date', 'target': 'target_binned_return_20',
        'features': 'feature_', 'horizon_days': 20,
        'train_end': '2024-12-31', 'dev_start': '2025-03-01',
        'dev_end': '2025-12-31', 'holdout_start': '2026-03-01',
        'source': 'crypto/v2.0/train.parquet',
    },
    'signals': {
        'files': ['numerai_public_datasets/signals/signals_v3.0_train.parquet',
                  'numerai_public_datasets/signals/signals_v3.0_validation.parquet'],
        'date': 'date', 'target': 'target_jupiter_60',
        'features': 'feature_', 'horizon_days': 60,
        'train_end': '2022-12-31', 'dev_start': '2023-04-01',
        'dev_end': '2024-12-31', 'holdout_start': '2025-04-01',
        'source': 'signals/v3.0/train.parquet + validation.parquet',
    },
}

OFFICIAL_FILES = {
    'crypto': ('crypto/v2.0/train.parquet', 'crypto/v2.0/live.parquet'),
    'signals': ('signals/v3.0/train.parquet', 'signals/v3.0/validation.parquet',
                'signals/v3.0/live.parquet'),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def source_inventory(competition: str, data_root: Path) -> dict:
    """Financial-data updater: validate and fingerprint registered inputs."""
    import pyarrow.parquet as pq

    spec = SPECS[competition]
    records = []
    for name in spec['files']:
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        file = pq.ParquetFile(path)
        names = file.schema_arrow.names
        required = {spec['date'], spec['target']}
        if not required.issubset(names):
            raise ValueError(f'{path}: missing {required - set(names)}')
        if not any(col.startswith(spec['features']) for col in names):
            raise ValueError(f'{path}: no registered feature columns')
        records.append({'path': str(path.resolve()), 'sha256': sha256(path),
                        'bytes': path.stat().st_size, 'rows': file.metadata.num_rows,
                        'mtime_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                        'schema': names})
    return {'competition': competition, 'official_dataset': spec['source'],
            'created_utc': datetime.now(timezone.utc).isoformat(), 'files': records,
            'status': 'inventory_only',
            'note': 'A file hash and schema check do not prove source freshness or point-in-time feature validity.'}


def refresh_status(data_root: Path) -> dict:
    """Report why the existing, broader financial refresh can or cannot start."""
    import fcntl

    lock_path = Path('/tmp/numerai_signals_daily_refresh.lock')
    space = shutil.disk_usage(data_root)
    with lock_path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            running = False
            fcntl.flock(handle, fcntl.LOCK_UN)
        except BlockingIOError:
            running = True
    return {'schema': 'toddler-finance-refresh-status/v1',
            'existing_refresh_running': running, 'free_gib': round(space.free / 2**30, 2),
            'existing_refresh_min_free_gib': 25,
            'ready_for_existing_refresh': not running and space.free >= 25 * 2**30,
            'note': 'The existing Numerai daily refresh owns writes to its databases and files.'}


def sync_official(competition: str, destination: Path) -> dict:
    """Financial updater: atomically mirror current official Parquet on claude-data."""
    import pyarrow.parquet as pq
    from numerapi import CryptoAPI, SignalsAPI

    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < 200 * 2**30:
        raise RuntimeError('official dataset mirror requires 200 GiB storage reserve')
    api = CryptoAPI() if competition == 'crypto' else SignalsAPI()
    available = set(api.list_datasets())
    records = []
    for dataset in OFFICIAL_FILES[competition]:
        if dataset not in available:
            raise ValueError(f'official API does not list {dataset}')
        final = destination / dataset
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.with_name(f'.{final.name}.{os.getpid()}.part')
        try:
            api.download_dataset(dataset, str(temporary))
            rows = pq.ParquetFile(temporary).metadata.num_rows
            new_hash = sha256(temporary)
            changed = not final.exists() or sha256(final) != new_hash
            if changed:
                temporary.replace(final)
            else:
                temporary.unlink()
            records.append({'dataset': dataset, 'path': str(final), 'sha256': new_hash,
                            'rows': rows, 'changed': changed})
        finally:
            temporary.unlink(missing_ok=True)
    result = {'schema': 'toddler-finance-official-sync/v1', 'competition': competition,
              'synced_utc': datetime.now(timezone.utc).isoformat(), 'files': records,
              'note': 'Training remains pinned to its prepared manifest until a new run is opened.'}
    atomic_json(destination / f'{competition}-latest-sync.json', result)
    return result


def _dates(values):
    import pandas as pd
    return pd.to_datetime(values, utc=True).dt.tz_convert(None)


def assign_splits(dates, competition: str):
    """Chronological partitions with a gap longer than the target horizon."""
    import numpy as np
    import pandas as pd

    spec = SPECS[competition]
    d = _dates(dates)
    train_end = pd.Timestamp(spec['train_end'])
    dev_start = pd.Timestamp(spec['dev_start'])
    dev_end = pd.Timestamp(spec['dev_end'])
    holdout_start = pd.Timestamp(spec['holdout_start'])
    if (dev_start - train_end).days <= spec['horizon_days']:
        raise ValueError('train/dev embargo is shorter than target horizon')
    if (holdout_start - dev_end).days <= spec['horizon_days']:
        raise ValueError('dev/holdout embargo is shorter than target horizon')
    label = np.full(len(d), 'embargo', dtype=object)
    label[d <= train_end] = 'train'
    label[(d >= dev_start) & (d <= dev_end)] = 'dev'
    label[d >= holdout_start] = 'holdout'
    return label


def prepare(competition: str, data_root: Path, out: Path) -> dict:
    """Financial preparer: feature allowlist, target maturity, time split."""
    import pandas as pd
    import pyarrow.parquet as pq

    inv = source_inventory(competition, data_root)
    spec = SPECS[competition]
    feature_cols = [x for x in inv['files'][0]['schema'] if x.startswith(spec['features'])]
    if competition == 'signals':
        # Country is categorical and requires explicit encoding; first trial is numeric only.
        feature_cols = [x for x in feature_cols if x != 'feature_country']
    columns = [spec['date'], spec['target'], *feature_cols]
    frames = [pq.read_table(data_root / name, columns=columns).to_pandas()
              for name in spec['files']]
    frame = pd.concat(frames, ignore_index=True)
    frame[spec['date']] = _dates(frame[spec['date']])
    frame = frame.loc[frame[spec['target']].notna()].copy()
    if frame.empty:
        raise ValueError('no matured targets')
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors='coerce').astype('float32')
    frame['_partition'] = assign_splits(frame[spec['date']], competition)
    counts = frame['_partition'].value_counts().to_dict()
    if any(counts.get(part, 0) == 0 for part in ('train', 'dev', 'holdout')):
        raise ValueError(f'empty required partition: {counts}')
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for part in ('train', 'dev', 'holdout'):
        dest = out / f'{competition}-{part}.parquet'
        subset = frame.loc[frame['_partition'] == part, columns].copy()
        temp = dest.with_name(f'.{dest.name}.{os.getpid()}.tmp')
        subset.to_parquet(temp, index=False)
        temp.replace(dest)
        artifacts[part] = {'path': str(dest), 'sha256': sha256(dest),
                           'rows': len(subset), 'first_date': str(subset[spec['date']].min().date()),
                           'last_date': str(subset[spec['date']].max().date())}
    manifest = {'schema': 'toddler-finance-prepared/v1', 'competition': competition,
                'source_inventory': inv, 'target': spec['target'], 'features': feature_cols,
                'split': {k: spec[k] for k in ('train_end', 'dev_start', 'dev_end', 'holdout_start', 'horizon_days')},
                'partitions': artifacts, 'holdout_policy': 'evaluate once after candidate and recipe are frozen'}
    atomic_json(out / f'{competition}-prepared.json', manifest)
    return manifest


def engineered_features(frame, competition: str):
    """Financial interpreter: predeclared rowwise features only."""
    import numpy as np

    added = []
    if competition == 'crypto':
        for stem in ('momentum', 'volatility', 'volume_ewa', 'close_ewa'):
            short, long = f'feature_{stem}_20d', f'feature_{stem}_60d'
            if short in frame and long in frame:
                name = f'derived_{stem}_20_minus_60'
                frame[name] = (frame[short] - frame[long]).replace([np.inf, -np.inf], np.nan).astype('float32')
                added.append(name)
    elif competition == 'signals':
        pairs = [('momentum_12w_factor', 'momentum_52w_factor'),
                 ('rsi_60d_country_ranknorm', 'rsi_130d_country_ranknorm')]
        for a, b in pairs:
            short, long = f'feature_{a}', f'feature_{b}'
            if short in frame and long in frame:
                name = f'derived_{a}_minus_{b}'
                frame[name] = (frame[short] - frame[long]).astype('float32')
                added.append(name)
    return frame, added


def per_date_corr(frame, predictions, target: str, date_col: str, horizon_days: int = 20) -> dict:
    """Research proxy: mean cross-sectional Spearman, never a payout score."""
    import numpy as np
    import pandas as pd

    scored = pd.DataFrame({'date': frame[date_col].to_numpy(), 'truth': frame[target].to_numpy(),
                           'prediction': np.asarray(predictions)})
    values = []
    dates = []
    for day, group in scored.groupby('date', sort=True):
        if len(group) >= 20 and group['truth'].nunique() > 1 and group['prediction'].nunique() > 1:
            values.append(group['truth'].corr(group['prediction'], method='spearman'))
            dates.append(day)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    ci = None
    block_length = None
    if len(values) >= 10:
        days = pd.to_datetime(dates)
        spacing = max(1, int(np.median(np.diff(days.to_numpy()).astype('timedelta64[D]').astype(int))))
        block_length = min(len(values), max(1, math.ceil(horizon_days / spacing)))
        rng = np.random.default_rng(20261011)
        starts = rng.integers(0, max(1, len(values) - block_length + 1),
                              size=(1000, math.ceil(len(values) / block_length)))
        offsets = np.arange(block_length)
        indices = (starts[:, :, None] + offsets).reshape(1000, -1)[:, :len(values)]
        means = values[indices].mean(axis=1)
        ci = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return {'mean_date_spearman': float(np.nanmean(values)) if len(values) else None,
            'date_count': int(len(values)), 'row_count': int(len(frame)),
            'positive_date_fraction': float(np.mean(values > 0)) if len(values) else None,
            'block_bootstrap_95pct_ci': ci, 'bootstrap_block_dates': block_length,
            'metric_scope': 'historical_proxy_not_official_Numerai_CORR_MMC_Alpha_MPC'}


def train(competition: str, out: Path, threads: int = 8) -> dict:
    """Competition specialist: dev-only model comparison, no holdout read."""
    import joblib
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    manifest = json.loads((out / f'{competition}-prepared.json').read_text())
    spec = SPECS[competition]
    train_df = pd.read_parquet(manifest['partitions']['train']['path'])
    dev_df = pd.read_parquet(manifest['partitions']['dev']['path'])
    if sha256(Path(manifest['partitions']['train']['path'])) != manifest['partitions']['train']['sha256']:
        raise ValueError('training data changed after preparation')
    if sha256(Path(manifest['partitions']['dev']['path'])) != manifest['partitions']['dev']['sha256']:
        raise ValueError('development data changed after preparation')
    features = manifest['features']
    train_df, added = engineered_features(train_df, competition)
    dev_df, _ = engineered_features(dev_df, competition)
    features = features + added
    x_train = train_df[features].replace([np.inf, -np.inf], np.nan)
    x_dev = dev_df[features].replace([np.inf, -np.inf], np.nan)
    y_train = train_df[spec['target']].astype('float32')
    candidates = {
        'ridge': make_pipeline(SimpleImputer(strategy='median', add_indicator=False), Ridge(alpha=1000.0)),
        'lightgbm': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.035, num_leaves=20,
                                       min_child_samples=200, colsample_bytree=0.8,
                                       reg_lambda=10.0, n_jobs=threads, verbosity=-1,
                                       deterministic=True, force_col_wise=True, random_state=20261011),
    }
    results = {}
    model_hashes = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_dev)
        results[name] = per_date_corr(dev_df, pred, spec['target'], spec['date'], spec['horizon_days'])
        model_path = out / f'{competition}-{name}.joblib'
        joblib.dump(model, model_path)
        model_hashes[name] = sha256(model_path)
    ranked = sorted(results, key=lambda name: results[name]['mean_date_spearman'] or -1e9, reverse=True)
    winner = ranked[0]
    report = {'schema': 'toddler-finance-dev/v1', 'competition': competition,
              'target': spec['target'], 'features': features, 'derived_features': added,
              'candidate_results': results, 'selected': winner,
              'model_sha256': model_hashes,
              'selection_rule': 'highest mean date Spearman on dev partition',
              'source_manifest_sha256': sha256(out / f'{competition}-prepared.json'),
              'claim': 'dev proxy only; no 95th-percentile claim'}
    rng = np.random.default_rng(20261011)
    shuffled = dev_df[spec['target']].to_numpy().copy()
    for _, index in dev_df.groupby(spec['date']).indices.items():
        shuffled[index] = rng.permutation(shuffled[index])
    control = dev_df[[spec['date']]].copy()
    control['_shuffled_target'] = shuffled
    winner_model = candidates[winner]
    report['within_date_shuffled_target_control'] = per_date_corr(
        control, winner_model.predict(x_dev), '_shuffled_target', spec['date'], spec['horizon_days'])
    atomic_json(out / f'{competition}-dev.json', report)
    return report


def evaluate_holdout(competition: str, out: Path) -> dict:
    """Single-use historical holdout gate after candidate selection."""
    import joblib
    import numpy as np
    import pandas as pd

    path = out / f'{competition}-holdout-result.json'
    if path.exists():
        raise FileExistsError(f'holdout already opened: {path}')
    manifest = json.loads((out / f'{competition}-prepared.json').read_text())
    report = json.loads((out / f'{competition}-dev.json').read_text())
    if sha256(out / f'{competition}-prepared.json') != report['source_manifest_sha256']:
        raise ValueError('prepared manifest changed after candidate selection')
    holdout = Path(manifest['partitions']['holdout']['path'])
    if sha256(holdout) != manifest['partitions']['holdout']['sha256']:
        raise ValueError('holdout data changed after preparation')
    frame = pd.read_parquet(holdout)
    frame, _ = engineered_features(frame, competition)
    model_path = out / f"{competition}-{report['selected']}.joblib"
    if sha256(model_path) != report['model_sha256'][report['selected']]:
        raise ValueError('selected model changed after development selection')
    model = joblib.load(model_path)
    predictions = model.predict(frame[report['features']].replace([np.inf, -np.inf], np.nan))
    result = {'schema': 'toddler-finance-holdout/v1', 'competition': competition,
              'selected': report['selected'], 'score': per_date_corr(frame, predictions,
              SPECS[competition]['target'], SPECS[competition]['date'],
              SPECS[competition]['horizon_days']),
              'evaluated_utc': datetime.now(timezone.utc).isoformat(),
              'model_sha256': sha256(model_path),
              'claim': 'historical holdout proxy only; live leaderboard percentile unknown'}
    # O_EXCL prevents an accidental second opening in concurrent runs.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return result


def llm_review(competition: str, out: Path, endpoint: str, model: str) -> dict:
    """Ask a local LLM for research hypotheses using schema/dev aggregates only."""
    if not endpoint.startswith('http://127.0.0.1:') and not endpoint.startswith('http://localhost:'):
        raise ValueError('LLM review endpoint must be loopback; private data must stay local')
    report = json.loads((out / f'{competition}-dev.json').read_text())
    payload = {'competition': competition, 'target': report['target'],
               'features': report['features'], 'dev_aggregate_only': report['candidate_results']}
    prompt = ('Propose at most three point-in-time-safe, testable financial feature hypotheses. '
              'For each, state formula, required timestamp/source, leakage risk, and a negative control. '
              'Do not infer holdout performance or claim a tournament percentile. '
              'Return concise JSON. Available schema and development aggregates: '
              + json.dumps(payload, sort_keys=True))
    body = {'model': model, 'messages': [
        {'role': 'system', 'content': 'You are a skeptical financial data reviewer. No investment advice.'},
        {'role': 'user', 'content': prompt}], 'temperature': 0.2, 'max_tokens': 600}
    request = Request(endpoint.rstrip('/') + '/chat/completions',
                      data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=120) as response:
        answer = json.load(response)
    result = {'schema': 'toddler-finance-llm-review/v1', 'competition': competition,
              'model': model, 'endpoint': endpoint, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
              'response': answer['choices'][0]['message'].get('content', ''),
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'unreviewed_hypotheses_not_training_labels'}
    atomic_json(out / f'{competition}-llm-review.json', result)
    return result


def score_proxies(competition: str, out: Path) -> dict:
    """Compute canonical Numerai CORR transform without claiming live scoring."""
    import joblib
    import numpy as np
    import pandas as pd
    from numerai_tools.scoring import numerai_corr

    manifest = json.loads((out / f'{competition}-prepared.json').read_text())
    development = json.loads((out / f'{competition}-dev.json').read_text())
    if sha256(out / f'{competition}-prepared.json') != development['source_manifest_sha256']:
        raise ValueError('prepared manifest changed')
    path = Path(manifest['partitions']['dev']['path'])
    if sha256(path) != manifest['partitions']['dev']['sha256']:
        raise ValueError('development data changed')
    frame = pd.read_parquet(path)
    frame, _ = engineered_features(frame, competition)
    result = {'schema': 'toddler-finance-score-proxies/v1', 'competition': competition,
              'score_definition': 'numerai_tools.scoring.numerai_corr per historical date; '
                                  'unneutralized; no MMC, MPC or live leaderboard', 'models': {}}
    for name in development['candidate_results']:
        model_path = out / f'{competition}-{name}.joblib'
        if sha256(model_path) != development['model_sha256'][name]:
            raise ValueError(f'model changed: {name}')
        model = joblib.load(model_path)
        predictions = model.predict(frame[development['features']].replace([np.inf, -np.inf], np.nan))
        scores = []
        for _, indexes in frame.groupby(SPECS[competition]['date'], sort=True).indices.items():
            truth = frame[SPECS[competition]['target']].iloc[indexes].reset_index(drop=True)
            if len(truth) < 20 or truth.nunique() < 2:
                continue
            pred = pd.DataFrame({'prediction': predictions[indexes]})
            score = float(numerai_corr(pred, truth)['prediction'])
            if np.isfinite(score):
                scores.append(score)
        result['models'][name] = {'mean_date_numerai_corr_proxy': float(np.mean(scores)),
                                  'date_count': len(scores), 'positive_date_fraction': float(np.mean(np.array(scores) > 0))}
    atomic_json(out / f'{competition}-score-proxies.json', result)
    return result


def predict_live(competition: str, out: Path, official: Path) -> dict:
    """Write a local, no-stake prospective forecast; never submit it."""
    import joblib
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    development = json.loads((out / f'{competition}-dev.json').read_text())
    model_path = out / f"{competition}-{development['selected']}.joblib"
    if sha256(model_path) != development['model_sha256'][development['selected']]:
        raise ValueError('selected model changed after development selection')
    live = official / ('crypto/v2.0/live.parquet' if competition == 'crypto'
                       else 'signals/v3.0/live.parquet')
    identifier = 'symbol' if competition == 'crypto' else 'numerai_ticker'
    columns = [identifier, SPECS[competition]['date'],
               *[x for x in development['features'] if x.startswith('feature_')]]
    frame = pq.read_table(live, columns=columns).to_pandas(ignore_metadata=True)
    if frame[identifier].isna().any() or frame[identifier].duplicated().any():
        raise ValueError('live identifiers missing or duplicated')
    frame[SPECS[competition]['date']] = _dates(frame[SPECS[competition]['date']])
    source_date = frame[SPECS[competition]['date']].max()
    if (pd.Timestamp.now(tz='UTC').tz_localize(None) - source_date).days > 7:
        raise ValueError(f'live source is older than seven days: {source_date}')
    frame, _ = engineered_features(frame, competition)
    model = joblib.load(model_path)
    raw = model.predict(frame[development['features']].replace([np.inf, -np.inf], np.nan))
    if not np.isfinite(raw).all():
        raise ValueError('non-finite live prediction')
    prediction = pd.Series(raw).rank(pct=True, method='average')
    result = pd.DataFrame({identifier: frame[identifier].to_numpy(), 'prediction': prediction.to_numpy()})
    source_hash = sha256(live)
    model_hash = sha256(model_path)
    dest = out / 'prospective' / competition / f'{source_date:%Y%m%d}-{source_hash[:12]}-{model_hash[:12]}.csv'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        audit_path = dest.with_suffix('.json')
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if audit.get('source_sha256') == source_hash and audit.get('model_sha256') == model_hash:
                return audit
        raise FileExistsError(f'prospective forecast already exists without matching audit: {dest}')
    with dest.open('x', encoding='utf-8') as stream:
        result.to_csv(stream, index=False)
    audit = {'schema': 'toddler-finance-prospective/v1', 'competition': competition,
             'created_utc': datetime.now(timezone.utc).isoformat(),
             'source_date': str(source_date.date()), 'source_path': str(live),
             'source_sha256': source_hash, 'model_sha256': model_hash,
             'rows': len(result), 'predictions_path': str(dest),
             'predictions_sha256': sha256(dest), 'submitted': False, 'stake': 0,
             'status': 'local_forecast_waiting_for_matured_targets'}
    atomic_json(dest.with_suffix('.json'), audit)
    return audit


def leaderboard_status(competition: str, out: Path, account: str) -> dict:
    """Read-only public account ranking; separate from a new model's quality."""
    from numerapi import CryptoAPI, SignalsAPI

    api = CryptoAPI() if competition == 'crypto' else SignalsAPI()
    entries = api.get_account_leaderboard(limit=2000, offset=0)
    if len(entries) >= 2000:
        raise RuntimeError('account leaderboard may be truncated; paginate before estimating percentile')
    matches = [row for row in entries if str(row.get('username', '')).casefold() == account.casefold()]
    if len(matches) != 1:
        raise ValueError(f'account {account!r} found {len(matches)} times in {len(entries)} entries')
    rank = int(matches[0]['rank'])
    total = len(entries)
    if rank < 1 or rank > total:
        raise ValueError(f'invalid rank {rank} of {total}')
    result = {'schema': 'toddler-finance-leaderboard/v1', 'competition': competition,
              'account': account, 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'rank': rank, 'entries': total, 'percentile_from_top': round(100 * (1 - (rank - 1) / total), 2),
              'top_5pct_rank_cutoff': math.ceil(0.05 * total),
              'note': 'Public account rank reflects the existing staked portfolio, not this research candidate.'}
    atomic_json(out / f'{competition}-account-leaderboard-latest.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inventory', 'refresh-status', 'sync-official',
                                          'prepare', 'train', 'score-proxies', 'predict-live', 'leaderboard',
                                          'evaluate-holdout', 'llm-review'))
    parser.add_argument('competition', choices=tuple(SPECS))
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8030/v1')
    parser.add_argument('--model', default='mom-live')
    parser.add_argument('--official-destination', type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument('--account', default='develuse')
    args = parser.parse_args()
    if args.action == 'inventory': result = source_inventory(args.competition, args.data_root)
    elif args.action == 'refresh-status': result = refresh_status(args.data_root)
    elif args.action == 'sync-official': result = sync_official(args.competition, args.official_destination)
    elif args.action == 'prepare': result = prepare(args.competition, args.data_root, args.out)
    elif args.action == 'train': result = train(args.competition, args.out, args.threads)
    elif args.action == 'score-proxies': result = score_proxies(args.competition, args.out)
    elif args.action == 'predict-live': result = predict_live(args.competition, args.out, args.official_destination)
    elif args.action == 'leaderboard': result = leaderboard_status(args.competition, args.out, args.account)
    elif args.action == 'evaluate-holdout': result = evaluate_holdout(args.competition, args.out)
    else: result = llm_review(args.competition, args.out, args.endpoint, args.model)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
