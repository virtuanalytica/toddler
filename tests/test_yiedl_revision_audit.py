from __future__ import annotations

from datetime import date

import polars as pl

from scripts.audit_yiedl_revisions import compare


def test_revisions_new_and_removed_keys_are_visible():
    old = pl.DataFrame({'date': [date(2026, 10, 1), date(2026, 10, 1)],
                        'symbol': ['BTC', 'ETH'], 'yiedl_pvm_mean': [1., 2.],
                        'yiedl_onchain_mean': [None, 1.], 'yiedl_sent_mean': [3., 4.]})
    new = pl.DataFrame({'date': [date(2026, 10, 1), date(2026, 10, 1)],
                        'symbol': ['BTC', 'SOL'], 'yiedl_pvm_mean': [1.1, 2.],
                        'yiedl_onchain_mean': [None, 1.], 'yiedl_sent_mean': [3., 4.]})
    result = compare(new, old, date(2026, 10, 1), date(2026, 10, 1))
    assert result['overlap_rows'] == 1
    assert result['new_keys_in_old_dates'] == 1
    assert result['removed_keys_in_old_dates'] == 1
    assert result['changed_values'] == {'yiedl_pvm_mean': 1, 'yiedl_onchain_mean': 0,
                                        'yiedl_sent_mean': 0}
    assert result['revision_detected']


def test_identical_snapshot_has_no_revision():
    frame = pl.DataFrame({'date': [date(2026, 10, 1)], 'symbol': ['BTC'],
                          'yiedl_pvm_mean': [1.], 'yiedl_onchain_mean': [None],
                          'yiedl_sent_mean': [2.]})
    result = compare(frame, frame, date(2026, 10, 1), date(2026, 10, 1))
    assert not result['revision_detected']
