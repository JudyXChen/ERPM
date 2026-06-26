"""Partial Rank Correlation Coefficient (PRCC).

Method:
  1. rank-transform all inputs and the output;
  2. for input i, linearly regress rank(x_i) and rank(y) on the ranks of all
     *other* inputs;
  3. PRCC_i = Pearson correlation of the two residual vectors;
  4. significance from a t-statistic with df = n - 2 - k (k = #partialled out).
  
"""

import numpy as np
from scipy import stats


def prcc(X, y):
    """PRCC of every column of X against y.

    Parameters
    ----------
    X : (n_samples, n_params) array of inputs (raw values; rank-transformed
        internally, so log vs linear scaling is irrelevant).
    y : (n_samples,) array of one output.  Rows where y is NaN are dropped.

    Returns
    -------
    rho  : (n_params,) PRCC in [-1, 1]; NaN for a degenerate column.
    pval : (n_params,) two-sided p-value for rho != 0.
    n_used : int, number of finite rows actually used.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[keep], y[keep]
    n, p = X.shape
    rho = np.full(p, np.nan)
    pval = np.full(p, np.nan)
    if n <= p + 2:
        return rho, pval, n

    Xr = np.column_stack([stats.rankdata(X[:, j]) for j in range(p)])
    yr = stats.rankdata(y)
    df = n - 2 - (p - 1)  # residual dof after partialling out the p-1 others

    for i in range(p):
        others = [j for j in range(p) if j != i]
        Z = np.column_stack([np.ones(n), Xr[:, others]])
        beta_x, *_ = np.linalg.lstsq(Z, Xr[:, i], rcond=None)
        beta_y, *_ = np.linalg.lstsq(Z, yr, rcond=None)
        res_x = Xr[:, i] - Z @ beta_x
        res_y = yr - Z @ beta_y
        sx, sy = res_x.std(), res_y.std()
        if sx == 0 or sy == 0:
            continue
        r = float(np.clip(np.corrcoef(res_x, res_y)[0, 1], -1.0, 1.0))
        rho[i] = r
        if df > 0 and abs(r) < 1.0:
            t = r * np.sqrt(df / (1.0 - r * r))
            pval[i] = float(2.0 * stats.t.sf(abs(t), df))
        else:
            pval[i] = 0.0 if abs(r) >= 1.0 else np.nan
    return rho, pval, n


def prcc_matrix(X, Y, qoi_names):
    """PRCC for several QoIs at once.

    X : (n, p) inputs; Y : dict qoi -> (n,) array.  Returns
    {qoi: {"rho": [...], "pval": [...], "n_used": int}} for each name in
    qoi_names that is present in Y.
    """
    out = {}
    for q in qoi_names:
        if q not in Y:
            continue
        rho, pval, n_used = prcc(X, Y[q])
        out[q] = {
            "rho": [float(v) for v in rho],
            "pval": [float(v) for v in pval],
            "n_used": int(n_used),
        }
    return out
