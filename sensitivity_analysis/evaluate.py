"""QoIs for one parameter sample.

Three tiers sharing one stable schema (keys not computed come back NaN):
  qoi_resting(p)     -- cheap: equilibrate gating at the clamped resting state,
                        then read the ER flux balance + open fractions.
  qoi_trajectory(p)  -- one free integration: ER Ca left, drain half-life, Ca_c.
  qoi_stimulus(p)    -- single glutamate pulse (EPSP+bAP); features of the
                        evoked cytosolic-Ca transient + per-pathway Ca flux.

At t=0 every channel is closed, so the drain only develops once the Markov
gating equilibrates; equilibrate_gating() pre-equilibrates gating with Ca_c and
Ca_ER clamped so dCa_ER/dt is read at the true resting balance. The stimulus
tier reuses that equilibrated state as the pre-stimulus baseline.
"""

import numpy as np
from scipy.integrate import solve_ivp

from model.parameters import default_parameters
from model.ode0d import build_system, rhs, _make_voltage
from model import stimulus

# Stimulus-response QoIs: features of the evoked Ca_cyto transient + the
# per-pathway Ca delivered to the cytosol over the response window.
STIM_QOI_NAMES = (
    "ca_cyto_peak",       # peak Ca_cyto above baseline (M)
    "ca_cyto_ttp",        # time from pulse to peak (s)
    "ca_cyto_fwhm",       # full-width at half-max of the transient (s)
    "ca_cyto_tau_decay",  # decay tau of the fall toward the plateau (s)
    "ca_cyto_auc",        # integral of (Ca_cyto - baseline) over window (M*s)
    "ca_cyto_plateau",    # sustained Ca_cyto above baseline at window end (M)
    "ip3r_peak_open",     # peak IP3R open fraction during the response
    "ryr_peak_open",      # peak RyR open fraction during the response
    "q_ipr",              # Ca delivered to cytosol via IP3R over window (M)
    "q_ryr",              # Ca delivered to cytosol via RyR over window (M)
    "q_serca",            # Ca pumped out of cytosol by SERCA over window (M)
    "q_soce",             # Ca entering cytosol via SOCE over window (M)
    "cicr_gain",          # ER-release share of total Ca influx (dimensionless)
)

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
) + STIM_QOI_NAMES

_TAU_CAP = 1.0e6  # s; cap the "stable" tail so NaN/inf don't poison SALib
_STIM_WINDOW = 0.050  # s; response window after the single glutamate pulse


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


# --- stimulus-response helpers ---

# Cytosolic-Ca source per reaction. Only the channels carrying meaningful Ca
# are mapped; gating-bind/buffer/lumped-decay terms fall through to "other"
# and are not integrated into a pathway QoI.
def _cac_group(name):
    if name == "IP3R_flux":
        return "IP3R"
    if name == "RyR_CICR":
        return "RyR"
    if name in ("S_x_x1", "S_x1_x2"):   # SERCA pump binding cytosolic Ca
        return "SERCA"
    if name.startswith("SOCE"):
        return "SOCE"
    if name in ("N_Ca_entry", "V_Ca_entry"):  # NMDAR + VSCC plasma-membrane influx
        return "PM_influx"
    return "other"


def _integrate_stimulus(compiled, y0, vfun, window, n_eval):
    """Integrate the stimulus response, restarting at each sharp V_m onset so
    BDF does not step over the EPSP/bAP kinks. Returns (t, Y)."""
    onsets = [o for o in stimulus.voltage_onsets([0.0]) if 0.0 < o < window]
    t_eval = np.linspace(0.0, window, n_eval)
    t0 = 0.0
    y = np.asarray(y0, dtype=float).copy()
    T, Y = [], []
    for b in sorted(set(onsets)) + [window]:
        mask = (t_eval > t0) & (t_eval <= b) if T else (t_eval >= t0) & (t_eval <= b)
        sol = solve_ivp(lambda t, yy: rhs(t, yy, compiled, vfun), (t0, b), y,
                        method="BDF", t_eval=t_eval[mask], rtol=1e-7, atol=1e-13)
        T.append(sol.t)
        Y.append(sol.y)
        y = sol.y[:, -1].copy()
        t0 = b
    return np.concatenate(T), np.concatenate(Y, axis=1)


def _fwhm(t, ca, base, amp, ipk):
    """Full-width at half-max of the transient; NaN if it never crosses back."""
    half = base + 0.5 * amp
    up = np.where(ca[:ipk + 1] <= half)[0]
    down = np.where(ca[ipk:] <= half)[0]
    if up.size == 0 or down.size == 0:
        return float("nan")
    iu = up[-1]  # last sample below half before the peak
    t_up = np.interp(half, [ca[iu], ca[iu + 1]], [t[iu], t[iu + 1]])
    idn = ipk + down[0]  # first sample below half after the peak
    t_dn = np.interp(half, [ca[idn], ca[idn - 1]], [t[idn], t[idn - 1]])
    return float(t_dn - t_up)


def _tau_decay(t, ca, floor, ipk):
    """Single-exp tau of the post-peak fall toward `floor` (the plateau), via a
    log-linear fit. NaN if too few points sit above the floor to fit."""
    td, yd = t[ipk:], ca[ipk:] - floor
    keep = yd > 0
    if keep.sum() < 5:
        return float("nan")
    slope = np.polyfit(td[keep], np.log(yd[keep]), 1)[0]
    return float(-1.0 / slope) if slope < 0 else float("nan")


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


def qoi_resting(p, *, eq=None):
    names, y_eq, compiled, idx, vfun = eq if eq is not None else equilibrate_gating(p)
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


def qoi_stimulus(p, *, eq=None, window=_STIM_WINDOW, n_eval=1001, tail=0.005):
    """Single glutamate pulse (EPSP+bAP) from the equilibrated resting state.
    Features of the evoked Ca_cyto transient + Ca delivered to the cytosol per
    pathway, integrated over `window`. `tail` is the trailing slice averaged
    for the sustained plateau level. `eq` reuses a shared equilibration."""
    names, y_eq, compiled, idx, _ = eq if eq is not None else equilibrate_gating(p)
    cac = idx["Ca_c"]
    glu_idx = idx.get("Glu")
    vfun = _make_voltage(stimulus.voltage(times=[0.0], V_rest=-65.0))

    y0 = y_eq.copy()
    if glu_idx is not None:
        y0[glu_idx] = stimulus.molecules_to_conc(stimulus.GLUT_BOLUS_MOLECULES, p)
    t, Y = _integrate_stimulus(compiled, y0, vfun, window, n_eval)

    ca = Y[cac]
    base = float(y_eq[cac])
    ipk = int(np.argmax(ca))
    peak, ttp = float(ca[ipk]), float(t[ipk])
    amp = peak - base
    plateau = float(np.mean(ca[t >= window - tail])) - base
    auc = float(np.trapezoid(ca - base, t))

    # Per-pathway cytosolic Ca flux: integrate the grouped dCa_c/dt
    # contributions over the window (reuses the same flux evaluation as the
    # ER-drain decomposition, retargeted to Ca_c).
    flux = {g: np.zeros(t.size) for g in ("IP3R", "RyR", "SERCA", "SOCE", "PM_influx")}
    for k in range(t.size):
        for nm, v in decompose_caer(Y[:, k], compiled, vfun, cac, t[k]).items():
            g = _cac_group(nm)
            if g in flux:
                flux[g][k] += v
    pos = lambda s: float(np.trapezoid(np.clip(s, 0.0, None), t))
    q_ipr, q_ryr = pos(flux["IP3R"]), pos(flux["RyR"])
    q_soce, q_pm = pos(flux["SOCE"]), pos(flux["PM_influx"])
    q_serca = float(-np.trapezoid(np.clip(flux["SERCA"], None, 0.0), t))  # uptake
    influx = q_ipr + q_ryr + q_soce + q_pm
    ip3r_open, ryr_open = open_fractions(Y, idx)
    return {
        "ca_cyto_peak": amp,
        "ca_cyto_ttp": ttp,
        "ca_cyto_fwhm": _fwhm(t, ca, base, amp, ipk),
        "ca_cyto_tau_decay": _tau_decay(t, ca, base + plateau, ipk),
        "ca_cyto_auc": auc,
        "ca_cyto_plateau": plateau,
        "ip3r_peak_open": float(np.nanmax(ip3r_open)),
        "ryr_peak_open": float(np.max(ryr_open)),
        "q_ipr": q_ipr,
        "q_ryr": q_ryr,
        "q_serca": q_serca,
        "q_soce": q_soce,
        "cicr_gain": (q_ipr + q_ryr) / influx if influx > 0 else float("nan"),
    }


def evaluate(overrides, *, trajectory=False, stim=False, t_max=1.0,
             stim_window=_STIM_WINDOW):
    """Full QoI dict for one sample. Any failure -> all-NaN (masked later)."""
    try:
        p = make_params(overrides)
        out = _nan_qois()
        eq = equilibrate_gating(p)  # shared by resting + stimulus tiers
        out.update(qoi_resting(p, eq=eq))
        if trajectory:
            out.update(qoi_trajectory(p, t_max=t_max))
        if stim:
            out.update(qoi_stimulus(p, eq=eq, window=stim_window))
        return out
    except Exception:  # extreme draws can stiffen the solver
        return _nan_qois()
