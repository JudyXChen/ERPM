"""Problem construction, log-space sampling, and parallel evaluation.

The SALib `problem` is built from the active groups.  Parameters are sampled in
log10 space (bounds = log10 of the lo/hi multiplicative range), so the sampler
sees a uniform box while the model sees log-uniform draws spanning orders of
magnitude.  `run_matrix` exponentiates each row back to physical values, applies
them as overrides, and evaluates the QoIs in parallel with joblib.
"""

import numpy as np
from joblib import Parallel, delayed

from model.parameters import default_parameters
from .config import active_groups, factor_for, RANGE_OVERRIDES, GROUPS
from .evaluate import evaluate, QOI_NAMES


def build_problem(include_soce=False, *, grouped=True, only_groups=None):
    """SALib problem dict with log10 bounds.

    grouped=True attaches a 'groups' entry (one Sobol/Morris index per
    mechanism); grouped=False omits it (per-parameter indices).

    only_groups (iterable of group names) restricts sampling to those
    mechanisms, holding every other parameter pinned at its nominal value.
    This is the lever for a *within-group* analysis: pass
    only_groups=["ryr", "setpoints"] with grouped=False to get one
    per-parameter Sobol/PRCC index for each RyR and setpoint parameter while
    the rest of the model stays fixed.
    """
    p0 = default_parameters()
    groups = active_groups(include_soce)
    if only_groups is not None:
        wanted = list(dict.fromkeys(only_groups))
        unknown = [g for g in wanted if g not in GROUPS]
        if unknown:
            raise ValueError(f"unknown group(s): {unknown}; "
                             f"choose from {sorted(GROUPS)}")
        missing = [g for g in wanted if g not in groups]
        if missing:
            raise ValueError(
                f"group(s) {missing} require include_soce=True (pass --soce)")
        groups = {g: groups[g] for g in wanted}
    names, bounds, group_of = [], [], []
    for g, params in groups.items():
        f = factor_for(g)
        for name in params:
            nominal = float(getattr(p0, name))
            if name in RANGE_OVERRIDES:
                lo, hi = RANGE_OVERRIDES[name]
            elif nominal > 0:
                lo, hi = nominal / f, nominal * f
            else:
                # non-positive nominal: fall back to a small symmetric box
                span = abs(nominal) if nominal else 1.0
                lo, hi = -span * f, span * f
            if lo <= 0:
                raise ValueError(
                    f"{name}: log sampling needs lo>0 (got {lo}); add an "
                    f"entry to RANGE_OVERRIDES")
            names.append(name)
            bounds.append([np.log10(lo), np.log10(hi)])
            group_of.append(g)

    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
    }
    if grouped:
        problem["groups"] = group_of
    return problem


def run_matrix(problem, X_log, *, include_soce=False, trajectory=False,
               n_jobs=-1, t_max=1.0, verbose=10):
    """Evaluate every row of X_log (log10 values) -> Y dict of arrays.

    Returns {qoi_name: np.ndarray(shape=[n_samples])}.  Rows that error come
    back as NaN; callers mask them before SALib analysis.
    """
    names = problem["names"]
    X = np.asarray(X_log, dtype=float)

    def _one(row):
        overrides = {nm: 10.0 ** row[i] for i, nm in enumerate(names)}
        return evaluate(overrides, include_soce=include_soce,
                        trajectory=trajectory, t_max=t_max)

    records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_one)(X[k]) for k in range(X.shape[0]))

    return {q: np.array([r[q] for r in records], dtype=float)
            for q in QOI_NAMES}


def failure_rate(Y, qoi):
    col = Y[qoi]
    return float(np.isnan(col).mean())


def clean_for_analysis(col):
    """SALib cannot handle NaN.  Replace with the column mean (a neutral fill);
    returns (filled_column, n_failed).  Caller should report n_failed."""
    col = np.asarray(col, dtype=float)
    bad = np.isnan(col)
    if bad.all():
        raise ValueError("all evaluations failed for this QoI")
    if bad.any():
        col = col.copy()
        col[bad] = np.nanmean(col)
    return col, int(bad.sum())
