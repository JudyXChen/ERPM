"""Evaluate the resting-baseline QoIs for one parameter sample.

Uses the equilibration / flux-decomposition machinery vendored in
sensitivity_analysis.resting so the SA measures the same resting 
quantities the manual one-at-a-time sweeps use.

Two tiers:
  qoi_resting(p)     -- cheap: equilibrate gating at the clamped resting state
                        and read the resting ER flux balance + open fractions.
                        Use this for the large grouped-Sobol sweep.
  qoi_trajectory(p)  -- one free integration (no clamp): how much ER Ca is left,
                        the drain half-life, and the emergent resting Ca_c.
                        Use for screening / within-group runs.

Every QoI is returned for both tiers (trajectory keys are NaN when only the
resting tier is requested) so the output schema is stable.
"""

import numpy as np

from model.parameters import default_parameters
from .resting import (
    equilibrate_gating,
    decompose_caer,
    grouped,
    open_fractions,
)

# Stable QoI ordering.  Downstream code indexes Y by these names.
QOI_NAMES = (
    "net_drain",      # resting net dCa_ER/dt (M/s); <0 means draining
    "abs_drain",      # |net_drain|; magnitude for variance attribution
    "tau_drain",      # Ca_ER / |net|  (s); large = stable rest
    "ip3r_open",      # resting IP3R open fraction
    "ryr_open",       # resting RyR open fraction
    "caer_frac_end",  # Ca_ER(t_max) / Ca_ER(0), free run
    "t_half",         # time for Ca_ER to fall to half (s); NaN if it never does
    "ca_c_end",       # emergent resting Ca_c at t_max (M)
)

_TAU_CAP = 1.0e6  # s; cap the "stable" tail so NaN/inf don't poison SALib


def make_params(overrides):
    """default_parameters() with `overrides` applied and derived values fixed.

    V_spine_L (liters) must track V_spine (um^3) or the N/(N_A*V) scalings break;
    we recompute it here so geometry can be sampled via V_spine alone.
    """
    p = default_parameters()
    for k, v in overrides.items():
        setattr(p, k, float(v))
    # keep the absolute spine volume consistent with the sampled um^3 value
    p.V_spine_L = float(p.V_spine) * 1e-15
    return p


def _nan_qois():
    return {k: float("nan") for k in QOI_NAMES}


def qoi_resting(p, *, include_soce=False):
    names, y_eq, compiled, idx, vfun = equilibrate_gating(
        p, include_soce=include_soce)
    caer = idx["Ca_ER"]
    g = grouped(decompose_caer(y_eq, compiled, vfun, caer))
    net = float(sum(g.values()))
    ip3r_open, ryr_open = open_fractions(y_eq, idx)
    ca_er0 = float(y_eq[caer])
    tau = ca_er0 / abs(net) if net != 0 else _TAU_CAP
    return {
        "net_drain": net,
        "abs_drain": abs(net),
        "tau_drain": min(tau, _TAU_CAP),
        "ip3r_open": float(ip3r_open),
        "ryr_open": float(ryr_open),
    }


def qoi_trajectory(p, *, include_soce=False, t_max=1.0):
    from model.ode0d import build_system, rhs, _make_voltage
    from scipy.integrate import solve_ivp

    names, y0, compiled, _ = build_system(p, include_soce=include_soce)
    idx = {n: i for i, n in enumerate(names)}
    caer, cac = idx["Ca_ER"], idx["Ca_c"]
    ca_er0 = float(y0[caer])
    vfun = _make_voltage(-65.0)
    sol = solve_ivp(lambda t, y: rhs(t, y, compiled, vfun), (0.0, t_max), y0,
                    method="BDF", rtol=1e-7, atol=1e-13,
                    t_eval=np.linspace(0.0, t_max, 1000))
    tr = sol.y[caer]
    below = np.where(tr <= 0.5 * ca_er0)[0]
    t_half = float(sol.t[below[0]]) if below.size else float("nan")
    return {
        "caer_frac_end": float(tr[-1] / ca_er0),
        "t_half": t_half,
        "ca_c_end": float(sol.y[cac][-1]),
    }


def evaluate(overrides, *, include_soce=False, trajectory=False, t_max=1.0):
    """Full QoI dict for one sample.  Failures -> all-NaN (masked later)."""
    out = _nan_qois()
    try:
        p = make_params(overrides)
        out.update(qoi_resting(p, include_soce=include_soce))
        if trajectory:
            out.update(qoi_trajectory(p, include_soce=include_soce,
                                      t_max=t_max))
    except Exception:
        # extreme draws can stiffen the solver; record NaN and move on
        return _nan_qois()
    return out
