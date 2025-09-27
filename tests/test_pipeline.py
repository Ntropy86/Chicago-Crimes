import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from l2_features import (
    _sanitize_and_cast_for_parquet,
    _assign_h3,
    _normalize_street,
    _cyclical_encode,
    H3_RESOLUTIONS,
)
from l3_multiscale import aggregate_daily_counts, aggregate_monthly_counts

# Test data
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'datetime': pd.to_datetime(['2023-01-01 12:00:00', '2023-01-02 13:00:00']),
        'latitude': [41.8781, 41.8782],
        'longitude': [-87.6298, -87.6299],
        'block_address': ['012XX W SOME ST', '045XX N OTHER AVE'],
        'arrest_made': [True, False],
        'beat_id': [123, 456],
        'hour_sin': [0.5, 0.6],
        'hour_cos': [0.866, 0.8],
    })

class TestSanitizer:
    def test_sanitize_and_cast_for_parquet(self, sample_df):
        df = sample_df.copy()
        result = _sanitize_and_cast_for_parquet(df, 9)

        # Check dtypes
        assert result['date'].dtype == 'datetime64[ns]'
        assert result['year'].dtype == 'int32'
        assert result['month'].dtype == 'int8'
        assert result['day_of_week'].dtype == 'int8'
        assert result['hour'].dtype == 'int8'
        assert result['hour_sin'].dtype == 'float64'
        assert result['hour_cos'].dtype == 'float64'
        assert result['arrest_made'].dtype == 'boolean'
        assert result['beat_id'].dtype == 'Int64'
        expected_h3 = f"h3_r{max(H3_RESOLUTIONS)}"
        assert expected_h3 in result.columns
        assert result[expected_h3].dtype == 'string'
        assert 'crime_category' in result.columns

    def test_normalize_street(self):
        series = pd.Series(['012XX W SOME ST', '045XX N OTHER AVE', '1000 E MAIN RD'])
        result = _normalize_street(series)
        expected = pd.Series(['W SOME ST', 'N OTHER AVE', 'E MAIN RD'])
        pd.testing.assert_series_equal(result, expected)

    def test_cyclical_encode(self):
        series = pd.Series([0, 6, 12, 18, 23])
        result = _cyclical_encode(series, 24, 'hour')
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
        assert len(result) == 5
        # Check known values
        assert np.isclose(result.loc[0, 'hour_sin'], 0.0)
        assert np.isclose(result.loc[0, 'hour_cos'], 1.0)

@pytest.mark.skipif('h3' not in sys.modules, reason="h3 library not available")
class TestH3Assignment:
    def test_assign_h3_success(self, sample_df):
        result = _assign_h3(sample_df, 9)
        assert len(result) == len(sample_df)
        assert result.dtype == 'string'
        # Should have valid H3 strings for valid coords
        assert pd.notna(result.iloc[0])
        assert pd.notna(result.iloc[1])

    def test_assign_h3_invalid_coords(self):
        df = pd.DataFrame({
            'latitude': [999, None],  # invalid
            'longitude': [-87.6298, -87.6299]
        })
        result = _assign_h3(df, 9)
        assert len(result) == 2
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])

class TestAggregator:
    def test_aggregate_daily(self):
        df = pd.DataFrame({
            'h3_r9': ['8928308280fffff', '8928308280fffff', '89283082807ffff'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02']),
            'arrest_made': [True, False, True]
        })
        result = aggregate_daily_counts(df, 'h3_r9', 'arrest_made')
        assert len(result) > 0
        assert 'h3_r9' in result.columns
        assert 'date' in result.columns
        assert 'n_crimes' in result.columns
        assert 'n_arrests' in result.columns

    def test_aggregate_monthly(self):
        df = pd.DataFrame({
            'h3_r9': ['8928308280fffff', '8928308280fffff', '89283082807ffff'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-15']),
            'arrest_made': [True, False, True]
        })
        daily = aggregate_daily_counts(df, 'h3_r9', 'arrest_made')
        result = aggregate_monthly_counts(daily, 'h3_r9')
        assert len(result) > 0
        assert 'h3_r9' in result.columns
        assert 'month' in result.columns
