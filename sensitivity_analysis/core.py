"""Problem construction, log-space sampling, parallel evaluation, and the
shared CLI/output plumbing used by the runners.

Parameters are sampled in log10 space (bounds = log10 of the lo/hi range), so
the sampler sees a uniform box while the model sees log-uniform draws.
`run_matrix` exponentiates each row, applies the overrides, and evaluates the
QoIs in parallel; SOCE is always active.
"""

import json
import pathlib

import numpy as np
from joblib import Parallel, delayed

from model.parameters import default_parameters
from .config import factor_for, RANGE_OVERRIDES, GROUPS
from .evaluate import evaluate, QOI_NAMES


def build_problem(*, grouped=True, only_groups=None):
    """SALib problem dict with log10 bounds.

    grouped=True attaches a 'groups' entry (one index per mechanism);
    grouped=False gives per-parameter indices. only_groups restricts sampling
    to those mechanisms (the rest pinned at nominal) -- the within-group lever.
    """
    p0 = default_parameters()
    groups = GROUPS
    if only_groups is not None:
        wanted = list(dict.fromkeys(only_groups))
        unknown = [g for g in wanted if g not in GROUPS]
        if unknown:
            raise ValueError(f"unknown group(s): {unknown}; "
                             f"choose from {sorted(GROUPS)}")
        groups = {g: GROUPS[g] for g in wanted}

    names, bounds, group_of = [], [], []
    for g, params in groups.items():
        f = factor_for(g)
        for name in params:
            nominal = float(getattr(p0, name))
            if name in RANGE_OVERRIDES:
                lo, hi = RANGE_OVERRIDES[name]
            elif nominal > 0:
                lo, hi = nominal / f, nominal * f
            else:  # non-positive nominal: small symmetric fallback box
                span = abs(nominal) or 1.0
                lo, hi = -span * f, span * f
            if lo <= 0:
                raise ValueError(f"{name}: log sampling needs lo>0 (got {lo}); "
                                 f"add a RANGE_OVERRIDES entry")
            names.append(name)
            bounds.append([np.log10(lo), np.log10(hi)])
            group_of.append(g)

    problem = {"num_vars": len(names), "names": names, "bounds": bounds}
    if grouped:
        problem["groups"] = group_of
    return problem


def run_matrix(problem, X_log, *, trajectory=False, stim=False, n_jobs=-1,
               t_max=1.0, stim_window=0.050, verbose=10):
    """Evaluate every row of X_log (log10 values) -> {qoi: array}.

    Rows that error come back as NaN; callers mask them before SALib analysis.
    """
    names = problem["names"]
    X = np.asarray(X_log, dtype=float)

    def _one(row):
        overrides = {nm: 10.0 ** row[i] for i, nm in enumerate(names)}
        return evaluate(overrides, trajectory=trajectory, stim=stim,
                        t_max=t_max, stim_window=stim_window)

    records = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_one)(X[k]) for k in range(X.shape[0]))
    return {q: np.array([r[q] for r in records], dtype=float) for q in QOI_NAMES}


def clean_for_analysis(col):
    """SALib cannot handle NaN: replace with the column mean (neutral fill).
    Returns (filled_column, n_failed)."""
    col = np.asarray(col, dtype=float)
    bad = np.isnan(col)
    if bad.all():
        raise ValueError("all evaluations failed for this QoI")
    if bad.any():
        col = col.copy()
        col[bad] = np.nanmean(col)
    return col, int(bad.sum())


def add_common_args(ap, *, default_out):
    """argparse options shared by every runner."""
    ap.add_argument("--trajectory", action="store_true",
                    help="also compute free-run QoIs (caer_frac_end / ca_c_end)")
    ap.add_argument("--stimulus", action="store_true",
                    help="also compute single-pulse stimulus-response QoIs "
                         "(Ca_cyto transient + per-pathway flux)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--t-max", type=float, default=1.0, dest="t_max",
                    help="free-integration window (s) for trajectory QoIs")
    ap.add_argument("--stim-window", type=float, default=0.050, dest="stim_window",
                    help="response window (s) after the glutamate pulse")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=default_out)


def save_run(out_dir, json_name, results, *, X=None, Y=None, plot=None):
    """Write the indices JSON (+ optional samples + plots) for a run."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / json_name, "w") as fh:
        json.dump(results, fh, indent=2)
    if X is not None and Y is not None:
        np.savez(out / "samples.npz", X=X, **{f"Y_{k}": v for k, v in Y.items()})
    if plot is None:
        print(f"\nData -> {out}/")
        return
    try:
        plot(out)
        print(f"\nPlots + data -> {out}/")
    except Exception as exc:  # plotting is optional
        print(f"\nData -> {out}/  (plot skipped: {exc})")
