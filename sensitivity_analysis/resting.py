"""Resting-state helpers for the 0D model.

At t=0 every channel sits closed, so the instantaneous ER flux is zero -- the
drain only develops once the Markov gating equilibrates to the resting Ca levels.
equilibrate_gating() therefore pre-equilibrates all gating species while holding
Ca_c and Ca_ER clamped, so dCa_ER/dt can be read off at the true resting balance.
"""

import numpy as np
from scipy.integrate import solve_ivp

from model.ode0d import build_system, rhs, _make_voltage


# Reactions are grouped into ER-flux sources by name prefix.
def _group(name):
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


def equilibrate_gating(p, *, include_soce=False, t_eq=2.0, v_rest=-65.0):
    """Integrate with Ca_c and Ca_ER frozen so gating reaches resting steady
    state. Returns (names, y_eq, compiled, idx, vfun) at the clamped state."""
    names, y0, compiled, _ = build_system(p, include_soce=include_soce)
    idx = {n: i for i, n in enumerate(names)}
    ca_c, ca_er = idx["Ca_c"], idx["Ca_ER"]
    vfun = _make_voltage(float(v_rest))

    frozen = (ca_c, ca_er)

    def rhs_frozen(t, y):
        d = rhs(t, y, compiled, vfun)
        for i in frozen:
            d[i] = 0.0
        return d

    sol = solve_ivp(rhs_frozen, (0.0, t_eq), y0, method="BDF",
                    rtol=1e-8, atol=1e-14, dense_output=False)
    y_eq = sol.y[:, -1].copy()
    # Restore the clamped values exactly (guard against tiny drift).
    y_eq[ca_c] = y0[ca_c]
    y_eq[ca_er] = y0[ca_er]
    return names, y_eq, compiled, idx, vfun


def decompose_caer(y, compiled, vfun, caer_idx, t=0.0):
    """Per-reaction contribution to dCa_ER/dt at state y (units M/s)."""
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
