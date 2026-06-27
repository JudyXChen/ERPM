"""Resting-baseline QoIs for one parameter sample (SOCE always active).

Two tiers sharing one stable schema (trajectory keys are NaN when not run):
  qoi_resting(p)     -- cheap: equilibrate gating at the clamped resting state,
                        then read the ER flux balance + open fractions.
  qoi_trajectory(p)  -- one free integration: ER Ca left, drain half-life, Ca_c.

At t=0 every channel is closed, so the drain only develops once the Markov
gating equilibrates; equilibrate_gating() pre-equilibrates gating with Ca_c and
Ca_ER clamped so dCa_ER/dt is read at the true resting balance.
"""

import numpy as np
from scipy.integrate import solve_ivp

from model.parameters import default_parameters
from model.ode0d import build_system, rhs, _make_voltage

# Stable QoI ordering. Downstream code indexes Y by these names.
QOI_NAMES = (
    "net_drain",      # resting net dCa_ER/dt (M/s); <0 means draining
    "abs_drain",      # |net_drain|; magnitude for variance attribution
    "tau_drain",      # Ca_ER / |net| (s); large = stable rest
    "ip3r_open",      # resting IP3R open fraction
    "ryr_open",       # resting RyR open fraction
    "caer_frac_end",  # Ca_ER(t_max) / Ca_ER(0), free run
    "t_half",         # time for Ca_ER to fall to half (s); NaN if it never does
    "ca_c_end",       # emergent resting Ca_c at t_max (M)
)

_TAU_CAP = 1.0e6  # s; cap the "stable" tail so NaN/inf don't poison SALib


# --- resting equilibration helpers ---

def _group(name):
    """Map a reaction name to its ER-flux source."""
    if name == "S_leak":
        return "SERCA_leak"
    if name.startswith("S_"):
        return "SERCA_pump"
    if name.startswith("IP3R"):
        return "IP3R"
    if name.startswith("RyR"):
        return "RyR"
    if name.startswith("SOCE"):
        return "SOCE"
    return "other"


def equilibrate_gating(p, *, t_eq=2.0, v_rest=-65.0):
    """Integrate with Ca_c and Ca_ER frozen so gating reaches resting steady
    state. Returns (names, y_eq, compiled, idx, vfun) at the clamped state."""
    names, y0, compiled, _ = build_system(p, include_soce=True)
    idx = {n: i for i, n in enumerate(names)}
    ca_c, ca_er = idx["Ca_c"], idx["Ca_ER"]
    vfun = _make_voltage(float(v_rest))

    def rhs_frozen(t, y):
        d = rhs(t, y, compiled, vfun)
        d[ca_c] = d[ca_er] = 0.0
        return d

    sol = solve_ivp(rhs_frozen, (0.0, t_eq), y0, method="BDF",
                    rtol=1e-8, atol=1e-14, dense_output=False)
    y_eq = sol.y[:, -1].copy()
    y_eq[ca_c], y_eq[ca_er] = y0[ca_c], y0[ca_er]  # undo any tiny drift
    return names, y_eq, compiled, idx, vfun


def decompose_caer(y, compiled, vfun, caer_idx, t=0.0):
    """Per-reaction contribution to dCa_ER/dt at state y (M/s)."""
    Vm = vfun(t)
    contrib = {}
    for nm, ff, fo, rf, ro, st in compiled:
        if caer_idx not in st:
            continue
        net = 0.0
        if ff is not None:
            net += ff(*(y[i] for i in fo), Vm)
        if rf is not None:
            net -= rf(*(y[i] for i in ro), Vm)
        contrib[nm] = st[caer_idx] * net
    return contrib


def grouped(contrib):
    g = {}
    for nm, v in contrib.items():
        g[_group(nm)] = g.get(_group(nm), 0.0) + v
    return g


def open_fractions(y, idx):
    """Resting open fraction of IP3R (IPR_110) and RyR (sum of O states)."""
    ip3r_open = y[idx["IPR_110"]] if "IPR_110" in idx else float("nan")
    ryr_open = sum(y[i] for n, i in idx.items() if n.startswith("R_O_"))
    return ip3r_open, ryr_open


# --- QoIs ---

def make_params(overrides):
    """default_parameters() with overrides applied; V_spine_L (liters) is kept
    in sync with the sampled V_spine (um^3) so the N/(N_A*V) scalings hold."""
    p = default_parameters()
    for k, v in overrides.items():
        setattr(p, k, float(v))
    p.V_spine_L = float(p.V_spine) * 1e-15
    return p


def _nan_qois():
    return {k: float("nan") for k in QOI_NAMES}


def qoi_resting(p):
    names, y_eq, compiled, idx, vfun = equilibrate_gating(p)
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


def qoi_trajectory(p, *, t_max=1.0):
    names, y0, compiled, _ = build_system(p, include_soce=True)
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


def evaluate(overrides, *, trajectory=False, t_max=1.0):
    """Full QoI dict for one sample. Any failure -> all-NaN (masked later)."""
    try:
        p = make_params(overrides)
        out = _nan_qois()
        out.update(qoi_resting(p))
        if trajectory:
            out.update(qoi_trajectory(p, t_max=t_max))
        return out
    except Exception:  # extreme draws can stiffen the solver
        return _nan_qois()
