from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from scripts.research_yiedl_crypto import lagged_join


def test_yiedl_join_cannot_see_recent_or_future_rows():
    official = pl.DataFrame({'date': [date(2025, 1, 3), date(2025, 1, 4)],
                             'symbol': ['BTC', 'BTC']})
    yiedl = pl.DataFrame({'date': [date(2025, 1, 1), date(2025, 1, 3)],
                          'symbol': ['BTC', 'BTC'],
                          'yiedl_pvm_mean': [1.0, 9.0],
                          'yiedl_onchain_mean': [2.0, 9.0],
                          'yiedl_sent_mean': [3.0, 9.0]})
    joined = lagged_join(official, yiedl)
    assert joined['source_date'].to_list() == [date(2025, 1, 1), date(2025, 1, 1)]
    assert joined['yiedl_pvm_mean'].to_list() == [1.0, 1.0]


def test_yiedl_lag_is_protocol_requirement():
    empty = pl.DataFrame({'date': [], 'symbol': []}, schema={'date': pl.Date, 'symbol': pl.String})
    with pytest.raises(ValueError, match='two days'):
        lagged_join(empty, empty, days=1)
