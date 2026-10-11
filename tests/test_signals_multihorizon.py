from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import json

from scripts.research_signals_multihorizon import blend_ranked


def test_blend_ranks_within_each_week():
    dates = pd.to_datetime(['2024-01-05'] * 3 + ['2024-01-12'] * 3)
    primary = np.array([1., 2., 3., 100., 200., 300.])
    auxiliary = np.array([3., 2., 1., 300., 200., 100.])
    assert blend_ranked(dates, primary, auxiliary).tolist() == pytest.approx([2/3] * 6)


def test_invalid_blend_weight_rejected():
    with pytest.raises(ValueError, match='weight'):
        blend_ranked(pd.to_datetime(['2024-01-05']), [1.], [1.], 1.1)


def test_daily_cycle_runs_optional_mixture_only_after_sync(tmp_path, monkeypatch):
    from scripts import finance_daily_cycle as cycle
    from scripts import research_signals_multihorizon as mixture

    out = tmp_path / 'runs'
    artifact = out / 'signals_multihorizon_v1'
    artifact.mkdir(parents=True)
    (artifact / 'experiment.json').write_text('{}')
    calls = []
    monkeypatch.setattr(cycle, 'refresh_status', lambda root: {})
    monkeypatch.setattr(cycle, 'sync_official', lambda competition, root: calls.append(('sync', competition)) or {})
    monkeypatch.setattr(cycle, 'predict_live', lambda competition, out, root: calls.append(('predict', competition)) or {})
    monkeypatch.setattr(cycle, 'leaderboard_status', lambda competition, out, account: {})
    monkeypatch.setattr(mixture, 'predict', lambda official, out: calls.append(('mixture', str(out))) or {})
    monkeypatch.setattr('sys.argv', ['finance_daily_cycle', '--out', str(out),
                                    '--official-destination', str(tmp_path / 'official')])
    cycle.main()
    assert calls == [('sync', 'crypto'), ('predict', 'crypto'),
                     ('sync', 'signals'), ('predict', 'signals'), ('mixture', str(artifact))]
    report = next((out / 'daily_cycles').glob('*.json'))
    status = json.loads(report.read_text())
    assert status['competitions']['signals']['local_multihorizon_prediction']['status'] == 'ok'
