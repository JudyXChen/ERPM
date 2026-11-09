"""Fit the resting state: Ca_c 100 nM, Ca_ER 150 uM, IP3 = IP3_basal (RyR on, SOCE off).

PM leaks are written in the free-pump form used by pmca_ncx.py. Each is the
Bartol 2015 balance k3*[Ca]/KM at 100 nM, times one shared factor that offsets
the lumped sink and resting VSCC entry. SERCA molecular rates are never changed.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.parameters import default_parameters
from model.ode0d import build_system, equilibrate_state, _make_voltage, rhs

CA_C, CA_ER = 100e-9, 150e-6
OUT = ROOT / 'results/rest_calibration'


def balance_leaks(p):
    """Free-pump leak rates that exactly balance each extruder at 100 nM."""
    return (p.k_P3 * CA_C * p.k_P1 / (p.k_P2 + p.k_P3),
            p.k_NCX3 * CA_C * p.k_NCX1 / (p.k_NCX2 + p.k_NCX3))


def rest(values):
    p = default_parameters()
    p.Glu_init = 0.0
    for key, value in values.items():
        setattr(p, key, value)
    names, y, compiled, _ = build_system(p)
    eq = equilibrate_state(y, compiled, _make_voltage(-65), names=names, max_time_s=4000)
    if not eq.converged:
        raise RuntimeError('rest failed to converge')
    return names, compiled, eq, p


def fit(overrides=None, tol=1e-7):
    overrides = overrides or {}
    base = default_parameters()
    for key, value in overrides.items():
        setattr(base, key, value)
    pmca, ncx = balance_leaks(base)
    scale, r15, leak = 1.0, base.plc_gq_r15_scale, base.k_S_leak_gq
    fraction = base.plc_gq_kf9 * CA_C / (base.plc_gq_kf9 * CA_C + base.plc_gq_kr9)
    previous = None
    for _ in range(25):
        values = {**overrides, 'k_P_leak': scale * pmca, 'k_NCX_leak': scale * ncx,
                  'plc_gq_r15_scale': r15, 'k_S_leak_gq': leak,
                  'plc_gq_plc_ca_fraction': fraction}
        names, compiled, eq, p = rest(values)
        state = dict(zip(names, eq.state))
        error = (state['Ca_c'] / CA_C - 1, state['Ca_ER'] / CA_ER - 1, state['IP3'] / base.IP3_basal - 1)
        if max(map(abs, error)) < tol:
            return values, names, compiled, eq, p
        # Secant for the leak factor; ER and IP3 are first order in their own terms.
        if previous is None:
            next_scale = scale * (1 - 0.5 * error[0])
        else:
            next_scale = scale - error[0] * (scale - previous[0]) / (error[0] - previous[1])
        previous = (scale, error[0])
        scale, r15, leak = next_scale, r15 / (1 + error[2]), leak * (1 + error[1])
    raise RuntimeError('rest calibration did not converge')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    values, names, compiled, eq, p = fit()
    index = {n: i for i, n in enumerate(names)}
    ions = p.N_A * p.V_spine_L
    flux = {}
    for name, ff, fo, rf, ro, st in compiled:
        net = (ff(*(eq.state[i] for i in fo), -65.0) if ff else 0.0)
        net -= (rf(*(eq.state[i] for i in ro), -65.0) if rf else 0.0)
        if index['Ca_c'] in st:
            flux[name] = float(net * st[index['Ca_c']] * ions)
    deriv = rhs(0.0, eq.state, compiled, _make_voltage(-65))
    summary = {'fitted': values,
               'rest': {'Ca_c_nM': eq.state[index['Ca_c']] * 1e9,
                        'Ca_ER_uM': eq.state[index['Ca_ER']] * 1e6,
                        'IP3_nM': eq.state[index['IP3']] * 1e9},
               'max_abs_derivative': float(np.abs(deriv).max()),
               'equilibration_s': eq.time_s,
               'cytosolic_flux_ions_s': flux}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('fitted', 'rest', 'equilibration_s')}, indent=2))


if __name__ == '__main__':
    main()
