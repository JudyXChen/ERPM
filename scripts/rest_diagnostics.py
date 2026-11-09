"""Diagnostics at the fitted rest. Nothing here changes model/parameters.py.

Default: refill, capture, SERCA, stimulus and IP3 directions (diagnostics.json).
--leak-buffer: refit rest with leak, buffer and k_d changes (leak_buffer_diagnosis.json).
--vscc-sweep: refit rest across VSCC counts (vscc_sweep.csv).
Outputs go to results/rest_calibration/. Nothing is plotted.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import solve_ivp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import stimulus
from model.parameters import default_parameters
from model.ode0d import build_system, equilibrate_state, _make_voltage, rhs
from scripts.calibrate_rest import fit

OUT = ROOT / 'results/rest_calibration/diagnostics.json'
LEAK_OUT = OUT.with_name('leak_buffer_diagnosis.json')
VSCC_OUT = OUT.with_name('vscc_sweep.csv')
PM_TERMS = {'P_bind', 'N_bind', 'P_leak', 'N_leak'}
VREST = _make_voltage(-65.0)
GATE_60 = 1 / (1 + (60e-6 / default_parameters().K_STIM1) ** 4)


def params(mods=None):
    p = default_parameters()
    p.Glu_init = 0.0
    for key, value in (mods or {}).items():
        setattr(p, key, value)
    return p


BASE = params()
NAMES0, Y00, COMP0, _ = build_system(BASE)
REST = dict(zip(NAMES0, equilibrate_state(Y00, COMP0, VREST, names=NAMES0, max_time_s=4000).state))
IONS = BASE.N_A * BASE.V_spine_L
IONS_ER = BASE.N_A * BASE.V_spine_L * BASE.V_ER / BASE.V_spine


def fluxes(names, comp, y):
    idx = {n: i for i, n in enumerate(names)}
    out = {}
    for name, ff, fo, rf, ro, st in comp:
        net = ff(*(y[i] for i in fo), -65.0) if ff else 0.0
        net -= rf(*(y[i] for i in ro), -65.0) if rf else 0.0
        out[name] = (st.get(idx['Ca_c'], 0.0) * net * IONS,
                     st.get(idx['Ca_ER'], 0.0) * net * IONS_ER)
    return out


def refill(soce=False, mods=None, gate=None, pin=None, route=0.0, drop=(), T=120.0):
    """ER 60 uM, IP3 clamped, other shared states at SOCE-off rest."""
    p = params(mods)
    names, _, comp, _ = build_system(p, include_soce=soce)
    comp = [c for c in comp if c[0] not in drop]
    idx = {n: i for i, n in enumerate(names)}
    y = np.array([REST.get(n, 0.0) for n in names])
    y[idx['Ca_ER']] = 60e-6
    if soce:
        y[idx['P_O_STIM1']] = pin if pin is not None else (
            gate if gate is not None else 1 / (1 + (150e-6 / p.K_STIM1) ** 4))
    if route:
        comp = [(c[0], c[1], c[2], c[3], c[4],
                 {idx['Ca_c']: 1 - route, idx['Ca_ER']: route * p.V_spine / p.V_ER})
                if c[0] == 'SOCE_Ca_entry' else c for c in comp]

    def f(t, x):
        d = rhs(t, x, comp, VREST)
        d[idx['IP3']] = 0.0
        if pin is not None:
            d[idx['P_O_STIM1']] = 0.0
        return d

    s = solve_ivp(f, (0, T), y, method='BDF', rtol=1e-7, atol=1e-13, dense_output=True)
    t = np.linspace(0, T, 2401)
    er = s.sol(t)[idx['Ca_ER']] * 1e6

    def cross(level):
        hit = np.flatnonzero(er >= level)
        return float(t[hit[0]]) if len(hit) else None

    return {'t_105_s': cross(105), 't_135_s': cross(135), 't_148_s': cross(148),
            'ER_end_uM': float(er[-1])}


def capture(j=3000.0):
    """Partition of a steady extra cytosolic source with ER held at 60 uM."""
    names, _, comp, _ = build_system(params())
    idx = {n: i for i, n in enumerate(names)}
    results = []
    for source in (0.0, j):
        y = np.array([REST[n] for n in names])
        y[idx['Ca_ER']] = 60e-6

        def f(t, x):
            d = rhs(t, x, comp, VREST)
            d[idx['Ca_c']] += source / IONS
            d[idx['Ca_ER']] = 0.0
            d[idx['IP3']] = 0.0
            return d

        s = solve_ivp(f, (0, 60), y, method='BDF', rtol=1e-8, atol=1e-13, t_eval=[60])
        results.append(fluxes(names, comp, s.y[:, -1]))
    base, more = results
    delta = {k: more[k][0] - base[k][0] for k in base}
    return {'SERCA': (delta['S_x_x1'] + delta['S_x1_x2']) / -2 * 2 / j,
            'PMCA_bind': -delta['P_bind'] / j, 'NCX_bind': -delta['N_bind'] / j,
            'lumped_decay': -delta['Ca_lumped_decay'] / j,
            'reduced_PM_leak': -(delta['P_leak'] + delta['N_leak']) / j}


def serca_cycle(ca, er, p):
    k = np.zeros((6, 6))
    for a, b, f, r in ((0, 1, p.k_x0x1 * ca, p.k_x1x0), (1, 2, p.k_x1x2 * ca, p.k_x2x1),
                       (2, 3, p.k_x2y2, p.k_y2x2), (3, 4, p.k_y2y1, p.k_y1y2 * er),
                       (4, 5, p.k_y1y0, p.k_y0y1 * er), (5, 0, p.k_y0x0, p.k_x0y0)):
        k[b, a] += f; k[a, a] -= f; k[a, b] += r; k[b, b] -= r
    k[-1, :] = 1.0
    v = np.linalg.solve(k, np.eye(6)[-1])
    return 2 * (p.k_x2y2 * v[2] - p.k_y2x2 * v[3])


def serca_metrics(p):
    vmax = serca_cycle(1e-2, 150e-6, p)
    ca = np.geomspace(1e-9, 1e-3, 6001)
    j = np.array([serca_cycle(c, 150e-6, p) for c in ca])
    at = lambda frac: ca[np.argmin(abs(j - frac * vmax))]
    return {'Vmax_Ca_s_pump': vmax, 'K05_uM': at(.5) * 1e6,
            'Hill': np.log10(81) / np.log10(at(.9) / at(.1)),
            'rest_Ca_s_pump': serca_cycle(100e-9, 150e-6, p),
            'rest_ions_s': serca_cycle(100e-9, 150e-6, p) * p.N_SERCA}


def serca_scenarios():
    out = {}
    for label, factor in (('current', 1.0), ('K05_0.40', .689 / .40), ('K05_0.25', .689 / .25)):
        p = params({'k_x0x1': BASE.k_x0x1 * factor, 'k_x1x2': BASE.k_x1x2 * factor})
        m = serca_metrics(p)
        channels = 0.0
        flux = fluxes(NAMES0, COMP0, np.array([REST[n] for n in NAMES0]))
        channels = -(flux['RyR_CICR'][1] + flux['IP3R_flux'][1])
        leak = (m['rest_ions_s'] - channels) / (150e-6 * IONS_ER)
        m.update(required_leak_s=leak, ER_tau_s=1 / (leak + channels / (150e-6 * IONS_ER)))
        out[label] = m
    # Literature-like target: slower cycle and higher affinity, ratios preserved.
    slow, tight = 0.3, .753 / .27
    p = params({'k_x0x1': BASE.k_x0x1 * tight, 'k_x1x2': BASE.k_x1x2 * tight,
                'k_x2y2': BASE.k_x2y2 * slow, 'k_y2x2': BASE.k_y2x2 * slow,
                'k_y0x0': BASE.k_y0x0 * slow, 'k_x0y0': BASE.k_x0y0 * slow})
    m = serca_metrics(p)
    leak = (m['rest_ions_s'] - channels) / (150e-6 * IONS_ER)
    m.update(required_leak_s=leak, ER_tau_s=1 / (leak + channels / (150e-6 * IONS_ER)))
    out['Vmax_7_K05_0.27'] = m
    return out


def thapsigargin(n_orai=10.0, tau=5.0, T_free=120.0, T=240.0):
    """Francis/Nelson protocol: no Orai entry and SERCA blocked, then add-back.

    PMCA/NCX inward leaks do not depend on Ca_ext in this model, so they stay on.
    """
    p = params({'N_Orai1': n_orai, 'tau_STIM1': tau})
    names, y0, comp, _ = build_system(p, include_soce=True)
    idx = {n: i for i, n in enumerate(names)}
    rest = equilibrate_state(y0, comp, VREST, names=names, max_time_s=4000).state
    blocked = [c for c in comp if c[0] != 'S_x2_y2']
    free = [c for c in blocked if c[0] != 'SOCE_Ca_entry']
    a = solve_ivp(lambda t, x: rhs(t, x, free, VREST), (0, T_free), rest, method='BDF',
                  rtol=1e-7, atol=1e-13)
    b = solve_ivp(lambda t, x: rhs(t, x, blocked, VREST), (T_free, T), a.y[:, -1], method='BDF',
                  rtol=1e-7, atol=1e-13)
    ca_a, ca_b = a.y[idx['Ca_c']] * 1e9, b.y[idx['Ca_c']] * 1e9
    return {'Ca_c_rest_nM': ca_a[0], 'TG_Ca_c_peak_nM': ca_a.max(), 'Ca_c_before_addback_nM': ca_a[-1],
            'addback_Ca_c_peak_nM': ca_b.max(), 'addback_Ca_c_end_nM': ca_b[-1],
            'addback_t50_s': float(b.t[np.argmax(ca_b >= ca_a[-1] + .5 * (ca_b.max() - ca_a[-1]))] - T_free),
            'ER_end_TG_uM': a.y[idx['Ca_ER'], -1] * 1e6, 'gate_end_TG': a.y[idx['P_O_STIM1'], -1]}


def stimulated(soce=False, mods=None, T=120.0, times=(0.0,)):
    p = params(mods)
    names, y0, comp, _ = build_system(p, include_soce=soce)
    idx = {n: i for i, n in enumerate(names)}
    eq = equilibrate_state(y0, comp, VREST, names=names, max_time_s=4000)
    v = _make_voltage(stimulus.voltage(times=list(times), V_rest=-65.0))
    cur = eq.state.copy()
    ts, ys = [], []
    cuts = sorted({0.0, T, *times, *(t0 + 2e-3 for t0 in times)})
    for a, b in zip(cuts[:-1], cuts[1:]):
        if a in times:
            cur[idx['Glu']] = stimulus.molecules_to_conc(500, p)
        s = solve_ivp(lambda t, x: rhs(t, x, comp, v), (a, b), cur, method='BDF',
                      rtol=1e-7, atol=1e-13)
        ts.append(s.t); ys.append(s.y); cur = s.y[:, -1]
    t = np.concatenate(ts); y = np.concatenate(ys, axis=1)
    er, ip3 = y[idx['Ca_ER']] * 1e6, y[idx['IP3']] * 1e9
    i, j = int(er.argmin()), int(ip3.argmax())
    half = np.flatnonzero((t > t[i]) & (er >= er[i] + .5 * (er[0] - er[i])))
    down = np.flatnonzero((t > t[j]) & (ip3 <= ip3[0] + .5 * (ip3[j] - ip3[0])))
    row = {'Ca_c_rest_nM': y[idx['Ca_c'], 0] * 1e9, 'Ca_c_peak_nM': y[idx['Ca_c']].max() * 1e9,
           'ER_rest_uM': er[0], 'ER_min_uM': er[i], 'ER_min_t_s': t[i],
           'ER_half_recovery_s': float(t[half[0]] - t[i]) if len(half) else None,
           'IP3_rest_nM': ip3[0], 'IP3_peak_nM': ip3[j], 'IP3_peak_t_s': t[j],
           'IP3_half_decay_s': float(t[down[0]] - t[j]) if len(down) else None,
           'catalytic_peak_molecules': y[idx['PLC_Ca_GqGTP']].max() * p.A_PM}
    if len(times) > 1:
        row['ER_at_last_release_uM'] = float(np.interp(times[-1], t, er))
    if soce:
        gate = y[idx['P_O_STIM1']]
        row.update(gate_rest=gate[0], gate_peak=gate.max(), gate_peak_t_s=t[gate.argmax()])
    return {k: float(v) if v is not None else None for k, v in row.items()}


def run_diagnostics():
    pre = {'gate': GATE_60}
    r15 = BASE.plc_gq_r15_scale
    fast_ca = {'plc_gq_kf9': BASE.plc_gq_kf9 * 100, 'plc_gq_kr9': BASE.plc_gq_kr9 * 100,
               'plc_gq_kf11': BASE.plc_gq_kf11 * 100, 'plc_gq_kr11': BASE.plc_gq_kr11 * 100}
    result = {
        'rest_fluxes_ions_s': {k: v for k, v in fluxes(
            NAMES0, COMP0, np.array([REST[n] for n in NAMES0])).items() if v[0] or v[1]},
        'refill': {
            'soce_off': refill(),
            'soce_gate_from_rest': refill(soce=True),
            'soce_preactivated': refill(soce=True, **pre),
            'soce_pinned_1': refill(soce=True, pin=1.0),
            'route_25pct_preactivated': refill(soce=True, route=.25, **pre),
            'leak_off': refill(drop=('S_leak',), T=60.0),
            'ip3r_flux_off': refill(drop=('IP3R_flux',)),
            'ryr_flux_off': refill(drop=('RyR_CICR',)),
        },
        'capture_fraction': capture(),
        'thapsigargin_addback': {
            'orai_10_tau_5': thapsigargin(),
            'orai_10_tau_0.02': thapsigargin(tau=.0192),
            'orai_50_tau_5': thapsigargin(n_orai=50.0),
            'orai_2_tau_5': thapsigargin(n_orai=2.0),
        },
        'serca': serca_scenarios(),
        'stimulus': {'soce_off': stimulated(), 'soce_on': stimulated(soce=True)},
        'train_10_pulses_every_5s': {
            'soce_off': stimulated(T=100.0, times=tuple(5.0 * k for k in range(10))),
            'soce_on': stimulated(soce=True, T=100.0, times=tuple(5.0 * k for k in range(10))),
        },
        'ip3_directions': {
            'current_cycle_closed': stimulated(T=40.0),
            'calizo_unrepaired': stimulated(mods={'plc_gq_kf11': 3.34e4, 'plc_gq_kr11': .00334}, T=40.0),
            'K10_x40_instead': stimulated(mods={'plc_gq_kf11': 3.34e4, 'plc_gq_kr11': .00334,
                                                'plc_gq_kf10': 1.68}, T=40.0),
            'Ca_rates_x100': stimulated(mods=fast_ca, T=40.0),
            'decay_1_per_s_r15_x8': stimulated(mods={'plc_gq_k16': 1.0, 'plc_gq_r15_scale': r15 * 8}, T=40.0),
            'decay_0.5_r15_x4': stimulated(mods={'plc_gq_k16': .5, 'plc_gq_r15_scale': r15 * 4}, T=40.0),
            'gq_div10_kact_x10': stimulated(mods={'plc_gq_total_density': BASE.plc_gq_total_density / 10,
                                                  'plc_gq_k_act': BASE.plc_gq_k_act * 10}, T=40.0),
            'kact_x5': stimulated(mods={'plc_gq_k_act': BASE.plc_gq_k_act * 5}, T=40.0),
            'gq_total_div10': stimulated(mods={'plc_gq_total_density': BASE.plc_gq_total_density / 10}, T=40.0),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))


def respond(values, names, compiled, eq, p, end=60.0):
    """Single-stimulus response summary after a rest refit."""
    idx = {n: i for i, n in enumerate(names)}
    cur = eq.state.copy()
    cur[idx['Glu']] = stimulus.molecules_to_conc(500, p)
    voltage = _make_voltage(stimulus.voltage(times=[0.0], V_rest=-65.0))
    t, y = [], []
    for a, b in ((0.0, 2e-3), (2e-3, 0.1), (0.1, 1.0), (1.0, end)):
        s = solve_ivp(lambda tt, x: rhs(tt, x, compiled, voltage), (a, b), cur, method='BDF',
                      rtol=1e-7, atol=1e-13, max_step=(b - a) / 20)
        t.append(s.t); y.append(s.y); cur = s.y[:, -1]
    t, y = np.concatenate(t), np.concatenate(y, axis=1)
    ca, er = y[idx['Ca_c']] * 1e6, y[idx['Ca_ER']] * 1e6
    opened = y[[i for n, i in idx.items() if n.startswith('R_O_')]].sum(0)
    gradient = y[idx['Ca_ER']] - y[idx['Ca_c']]
    ryr = p.k_RyR * opened * gradient * p.N_RyR
    ip3r = p.k_IP3R_flux * y[idx['IPR_110']] ** 3 * gradient * p.N_IP3R
    peak = int(ca.argmax())
    half = np.flatnonzero((t > t[peak]) & (ca <= ca[0] + .5 * (ca[peak] - ca[0])))
    low = int(er.argmin())
    half_er = np.flatnonzero((t > t[low]) & (er >= er[low] + .5 * (er[0] - er[low])))
    first = t <= 1.0
    return {'N_VSCC': p.N_VSCC, 'Ca_rest_nM': ca[0] * 1e3, 'Ca_peak_uM': ca[peak],
            'rest_over_peak_pct': 100 * ca[0] / ca[peak], 'Ca_peak_ms': t[peak] * 1e3,
            'Ca_half_decay_ms': (t[half[0]] - t[peak]) * 1e3 if len(half) else np.nan,
            'RyR_rest_ions_s': ryr[0], 'RyR_peak_Po': opened.max(),
            'IP3R_rest_ions_s': ip3r[0], 'IP3R_peak_ions_s': ip3r.max(),
            'RyR_excess_1s_ions': np.trapezoid((ryr - ryr[0])[first], t[first]),
            'IP3R_excess_1s_ions': np.trapezoid((ip3r - ip3r[0])[first], t[first]),
            'ER_min_uM': er[low], 'ER_drop_pct': 100 * (1 - er[low] / er[0]),
            'ER_min_ms': t[low] * 1e3,
            'ER_half_recovery_s': t[half_er[0]] - t[low] if len(half_er) else np.nan,
            'IP3_peak_nM': y[idx['IP3']].max() * 1e9, 'k_P_leak': values['k_P_leak'],
            'k_NCX_leak': values['k_NCX_leak'], 'k_S_leak_gq': values['k_S_leak_gq'],
            'r15_scale': values['plc_gq_r15_scale'],
            'plc_ca_fraction': values['plc_gq_plc_ca_fraction']}


def vscc_sweep(counts):
    rows = [respond(*fit({'N_VSCC': float(n)})) for n in counts]
    VSCC_OUT.parent.mkdir(parents=True, exist_ok=True)
    with VSCC_OUT.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print({k: float(f'{v:.4g}') for k, v in row.items()}, flush=True)


def rest_cytosolic_fluxes(names, compiled, y, p):
    idx = {n: i for i, n in enumerate(names)}
    ions = p.N_A * p.V_spine_L
    out = {}
    for name, ff, fo, rf, ro, st in compiled:
        if idx['Ca_c'] in st:
            net = ff(*(y[i] for i in fo), -65.0) if ff else 0.0
            net -= rf(*(y[i] for i in ro), -65.0) if rf else 0.0
            value = float(net * st[idx['Ca_c']] * ions)
            if abs(value) > 1.0:
                out[name] = value
    return out


def reversed_pm_sign():
    """Flip the cytosolic sign of the PMCA/NCX pump and leak terms from fitted rest."""
    p = params()
    names, y0, compiled, _ = build_system(p)
    idx = {n: i for i, n in enumerate(names)}
    rest = equilibrate_state(y0, compiled, VREST, names=names, max_time_s=4000).state
    flipped = [(n, ff, fo, rf, ro, {k: (-v if k == idx['Ca_c'] and n in PM_TERMS else v)
                                    for k, v in st.items()})
               for n, ff, fo, rf, ro, st in compiled]
    out = {}
    for label, voltage, glu, stop in (
            ('no_stimulus', VREST, 0.0, 1.0),
            ('stimulus', _make_voltage(stimulus.voltage(times=[0.0], V_rest=-65.0)),
             stimulus.molecules_to_conc(500, p), 0.2)):
        y = rest.copy()
        y[idx['Glu']] = glu

        def limit(_t, x):
            return x[idx['Ca_c']] - 1e-3
        limit.terminal = True

        s = solve_ivp(lambda t, x: rhs(t, x, flipped, voltage), (0, stop), y, method='BDF',
                      rtol=1e-6, atol=1e-13, events=limit)
        ca = s.y[idx['Ca_c']]
        out[label] = {'Ca_start_nM': float(ca[0] * 1e9), 'Ca_end_nM': float(ca[-1] * 1e9),
                      'Ca_min_nM': float(ca.min() * 1e9), 'Ca_max_nM': float(ca.max() * 1e9),
                      'end_time_s': float(s.t[-1]), 'status': s.message}
    return out


def constant_leak(values, names, compiled, eq, p):
    """Replace the free-pump PM leaks by constant influx at their resting value."""
    changed = []
    for name, ff, fo, rf, ro, st in compiled:
        if name in ('P_leak', 'N_leak'):
            rate = ff(*(eq.state[i] for i in fo), -65.0)
            changed.append((name, (lambda _v, r=rate: r), (), None, (), st))
        else:
            changed.append((name, ff, fo, rf, ro, st))
    return respond(values, names, changed, eq, p, end=40.0)


def leak_buffer():
    p = default_parameters()
    result = {'reversed_pm_sign': reversed_pm_sign(), 'knockouts': {}}
    baseline = fit()
    values, names, compiled, eq, q = baseline
    result['rest_cytosolic_fluxes_ions_s'] = rest_cytosolic_fluxes(names, compiled, eq.state, q)
    result['constant_PM_leak'] = constant_leak(*baseline)
    cases = {'baseline': {}, 'k_d_0': {'k_d': 0.0}, 'k_d_half': {'k_d': p.k_d / 2},
             'k_d_double': {'k_d': p.k_d * 2}, 'Bm_0': {'Bm_total': 0.0}, 'Bf_0': {'Bf_total': 0.0},
             'Bm_Bf_double': {'Bm_total': p.Bm_total * 2, 'Bf_total': p.Bf_total * 2}}
    for label, overrides in cases.items():
        result['knockouts'][label] = respond(*fit(overrides), end=40.0)
        print(label, result['knockouts'][label], flush=True)
    LEAK_OUT.parent.mkdir(parents=True, exist_ok=True)
    LEAK_OUT.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({k: result[k] for k in ('reversed_pm_sign', 'constant_PM_leak',
                                             'rest_cytosolic_fluxes_ions_s')},
                     indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--leak-buffer', action='store_true')
    ap.add_argument('--vscc-sweep', type=float, nargs='*', metavar='N',
                    help='VSCC counts (default 1 5 6 7 8 9 14)')
    args = ap.parse_args()
    if args.vscc_sweep is not None:
        vscc_sweep(args.vscc_sweep or [1, 5, 6, 7, 8, 9, 14])
    elif args.leak_buffer:
        leak_buffer()
    else:
        run_diagnostics()


if __name__ == '__main__':
    main()
