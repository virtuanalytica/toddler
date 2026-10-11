from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research_signals_multihorizon import blend_ranked


def test_blend_ranks_within_each_week():
    dates = pd.to_datetime(['2024-01-05'] * 3 + ['2024-01-12'] * 3)
    primary = np.array([1., 2., 3., 100., 200., 300.])
    auxiliary = np.array([3., 2., 1., 300., 200., 100.])
    assert blend_ranked(dates, primary, auxiliary).tolist() == pytest.approx([2/3] * 6)


def test_invalid_blend_weight_rejected():
    with pytest.raises(ValueError, match='weight'):
        blend_ranked(pd.to_datetime(['2024-01-05']), [1.], [1.], 1.1)
