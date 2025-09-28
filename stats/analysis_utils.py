"""Reusable statistical helpers for Chicago crime analytics notebooks and apps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest
from statsmodels.stats.proportion import proportions_ztest


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
L3_BASE = DATA_DIR / "l3"
L2_BASE = DATA_DIR / "l2"


def _normalize_months(months: int | Iterable[int]) -> Tuple[int, ...]:
    if isinstance(months, (list, tuple, set)):
        return tuple(sorted(int(m) for m in months))
    return (int(months),)


def load_l3_partition(res: int, year: int, months: int | Iterable[int]) -> pd.DataFrame:
    """Load one or more L3 parquet partitions for a given resolution/year/months."""
    months_tuple = _normalize_months(months)
    frames: list[pd.DataFrame] = []
    for month in months_tuple:
        path = (
            L3_BASE
            / f"res={res}"
            / f"year={year}"
            / f"month={month:02d}"
            / f"l3-aggregates-{res}-{year}-{month:02d}.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing L3 partition: {path}")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_l2_partition(year: int, months: int | Iterable[int]) -> pd.DataFrame:
    """Load L2 feature partitions for the requested period."""
    months_tuple = _normalize_months(months)
    frames: list[pd.DataFrame] = []
    for month in months_tuple:
        path = L2_BASE / f"year={year}" / f"month={month:02d}" / f"features-{year}-{month:02d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing L2 partition: {path}")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_arrest_rate(df: pd.DataFrame, arrests_col: str = "n_arrests", incidents_col: str = "n_crimes") -> pd.Series:
    """Return arrest rate with safe division."""
    numer = df[arrests_col]
    denom = df[incidents_col].replace({0: np.nan})
    return (numer / denom).astype(float)


def build_daily_series(df: pd.DataFrame, date_col: str = "date", incidents_col: str = "n_crimes", arrests_col: str = "n_arrests") -> pd.DataFrame:
    """Aggregate a frame to daily incidents/arrests."""
    working = df.copy()
    working[date_col] = pd.to_datetime(working[date_col])
    grouped = working.groupby(working[date_col].dt.floor("D"))[[incidents_col, arrests_col]].sum().rename_axis("date").reset_index()
    grouped.rename(columns={incidents_col: "incidents", arrests_col: "arrests"}, inplace=True)
    return grouped


def extract_group_samples(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    groups: Sequence[str | int],
    dropna: bool = True,
) -> Mapping[str | int, pd.Series]:
    """Return a mapping of group values to sample series."""
    samples: dict[str | int, pd.Series] = {}
    for group in groups:
        series = df.loc[df[group_col] == group, value_col]
        samples[group] = series.dropna() if dropna else series
    return samples


def normality_test(series: pd.Series, *, method: str = "auto") -> Tuple[str, float, float]:
    """Run a normality test and return (method, stat, p)."""
    cleaned = series.dropna()
    if cleaned.empty:
        raise ValueError("Series has no non-null values for normality test.")
    if method == "auto":
        method = "shapiro" if len(cleaned) <= 5000 else "dagostino"
    if method == "shapiro":
        stat, p = stats.shapiro(cleaned)
    elif method == "dagostino":
        stat, p = stats.normaltest(cleaned)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return method, float(stat), float(p)


def variance_test(group1: pd.Series, group2: pd.Series, *, center: str = "median") -> Tuple[float, float]:
    """Levene's test for equal variances."""
    g1 = group1.dropna()
    g2 = group2.dropna()
    stat, p = stats.levene(g1, g2, center=center)
    return float(stat), float(p)


def cohens_d(group1: Sequence[float], group2: Sequence[float]) -> float:
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    diff = g1.mean() - g2.mean()
    pooled = np.sqrt(((g1.std(ddof=1) ** 2) + (g2.std(ddof=1) ** 2)) / 2)
    return float(diff / pooled) if pooled else float("nan")


def welch_ttest(group1: Sequence[float], group2: Sequence[float]) -> dict:
    stat, p = stats.ttest_ind(group1, group2, equal_var=False, nan_policy="omit")
    return {
        "test": "Welch t-test",
        "t_stat": float(stat),
        "p_value": float(p),
        "effect_size_d": cohens_d(group1, group2),
    }


def mann_whitney(group1: Sequence[float], group2: Sequence[float]) -> dict:
    stat, p = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    return {"test": "Mann-Whitney U", "u_stat": float(stat), "p_value": float(p)}


def proportion_ztest(success1: int, total1: int, success2: int, total2: int) -> dict:
    count = np.array([success1, success2])
    nobs = np.array([total1, total2])
    stat, p = proportions_ztest(count, nobs)
    diff = count[0] / nobs[0] - count[1] / nobs[1]
    se = np.sqrt((count[0] / nobs[0]) * (1 - count[0] / nobs[0]) / nobs[0] + (count[1] / nobs[1]) * (1 - count[1] / nobs[1]) / nobs[1])
    ci = (diff - 1.96 * se, diff + 1.96 * se)
    return {
        "test": "Two-proportion z-test",
        "z_stat": float(stat),
        "p_value": float(p),
        "diff": float(diff),
        "ci95": tuple(float(x) for x in ci),
    }


def correlation_tests(x: Sequence[float], y: Sequence[float]) -> dict:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    x_clean, y_clean = x_arr[mask], y_arr[mask]
    pearson = stats.pearsonr(x_clean, y_clean)
    spearman = stats.spearmanr(x_clean, y_clean)
    kendall = stats.kendalltau(x_clean, y_clean)
    return {
        "pearson_r": (float(pearson.statistic), float(pearson.pvalue)),
        "spearman_rho": (float(spearman.statistic), float(spearman.pvalue)),
        "kendall_tau": (float(kendall.statistic), float(kendall.pvalue)),
    }


def fdr_correction(p_values: Sequence[float], alpha: float = 0.05, method: str = "fdr_bh") -> dict:
    rejected, q_values, _, _ = multitest.multipletests(p_values, alpha=alpha, method=method)
    return {
        "method": method,
        "alpha": alpha,
        "rejected": list(map(bool, rejected)),
        "q_values": [float(q) for q in q_values],
    }


@dataclass
class TestReport:
    question: str
    hypothesis_null: str
    hypothesis_alt: str
    test_name: str
    statistic: float
    p_value: float
    effect_size: float | None = None
    notes: str | None = None

    def to_dict(self) -> dict:
        payload = {
            "question": self.question,
            "H0": self.hypothesis_null,
            "H1": self.hypothesis_alt,
            "test": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
        }
        if self.effect_size is not None:
            payload["effect_size"] = self.effect_size
        if self.notes:
            payload["notes"] = self.notes
        return payload
