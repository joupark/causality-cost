"""Cluster-robust inference for NOIZEUS analyses.

NOIZEUS reuses the same 30 sentences across every noise type (and every
SNR): a 210-utterance "sample" is really 30 sentences measured 7 times
each. Residuals for the same sentence are correlated (confirmed
empirically: sentence identity explains 41% of the noise-type-detrended
variance of B_c, one-way ANOVA F=4.27, p=7e-10), so treating all 210 rows
as independent understates every standard error. This module reports
cluster-robust (CR1, cluster = sentence) inference alongside the naive
i.i.d. inference used elsewhere, and is the version quoted as the
paper's confirmatory statistic wherever the two disagree materially.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def cluster_slope(x, y, clusters):
    """OLS slope of y on x with cluster-robust (CR1) SE.
    Returns dict with slope, se, t, p (df = n_clusters-1), r, n, n_clusters.
    Also returns the naive (i.i.d.) slope/r/p for comparison."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    clusters = np.asarray(clusters)
    X = sm.add_constant(x)
    n_clusters = len(np.unique(clusters))
    fit = sm.OLS(y, X).fit(cov_type='cluster',
                            cov_kwds={'groups': clusters, 'df_correction': True})
    slope = fit.params[1]
    se = fit.bse[1]
    df = max(n_clusters - 1, 1)
    tval = slope / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(tval), df) if se > 0 else np.nan
    naive = stats.linregress(x, y)
    r2 = fit.rsquared
    return dict(slope=slope, se=se, t=tval, p_cluster=p, df=df,
                n=len(x), n_clusters=n_clusters, r2=r2,
                slope_naive=naive.slope, r_naive=naive.rvalue, p_naive=naive.pvalue)


def holm_bonferroni(pvals, labels=None):
    """Holm-Bonferroni step-down correction. Returns list of (label, p_raw,
    p_adj, reject_at_0.05) sorted by raw p ascending."""
    labels = labels if labels is not None else list(range(len(pvals)))
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    out = [(labels[i], pvals[i], adj[i], adj[i] < 0.05) for i in range(m)]
    return sorted(out, key=lambda r: r[1])


def paired_monotone_test(df, value_col, level_col, unit_col, level_order):
    """Wilcoxon signed-rank test between consecutive levels (e.g. context
    C), paired by unit (utterance/sentence), to test monotonicity formally
    rather than by comparing means only."""
    out = []
    piv = df.pivot_table(index=unit_col, columns=level_col, values=value_col)
    for a, b in zip(level_order[:-1], level_order[1:]):
        d = (piv[b] - piv[a]).dropna()
        if len(d) < 4 or np.all(d.values == 0):
            out.append(dict(a=a, b=b, n=len(d), stat=np.nan, p=np.nan,
                             median_diff=float(d.median()) if len(d) else np.nan))
            continue
        stat, p = stats.wilcoxon(d, alternative='greater')
        out.append(dict(a=a, b=b, n=len(d), stat=stat, p=p,
                         median_diff=float(d.median())))
    return out
