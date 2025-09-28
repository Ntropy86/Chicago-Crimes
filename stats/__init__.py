"""Statistical helper utilities for Chicago Crime analytics."""

from .analysis_utils import (
    DATA_DIR,
    load_l3_partition,
    load_l2_partition,
    compute_arrest_rate,
    build_daily_series,
    extract_group_samples,
    normality_test,
    variance_test,
    cohens_d,
    welch_ttest,
    mann_whitney,
    proportion_ztest,
    correlation_tests,
    fdr_correction,
)

__all__ = [
    'DATA_DIR',
    'load_l3_partition',
    'load_l2_partition',
    'compute_arrest_rate',
    'build_daily_series',
    'extract_group_samples',
    'normality_test',
    'variance_test',
    'cohens_d',
    'welch_ttest',
    'mann_whitney',
    'proportion_ztest',
    'correlation_tests',
    'fdr_correction',
]
