"""Stimulus
"""

import sympy as sym


# Default single-bolus size (molecules) released at the PSD.
GLUT_BOLUS_MOLECULES = 500.0


# ---------------------------------------------------------------------------
# 1. Glutamate release train
# ---------------------------------------------------------------------------
def molecules_to_conc(n_molecules, p):
    """Spine concentration (M) for ``n_molecules`` released into V_spine."""
    return float(n_molecules) / (p.N_A * p.V_spine_L)


def release_times(frequency_hz=1.0, n_pulses=1, t_stim=0.0):
    """Glutamate release times (s): ``t_stim, t_stim+1/f, ...`` (``n_pulses``).
    """
    dt = 1.0 / float(frequency_hz)
    return [t_stim + k * dt for k in range(int(n_pulses))] 


# ---------------------------------------------------------------------------
# 2. Voltage: V_rest + EPSP + bAP centred on release time
# ---------------------------------------------------------------------------
def voltage(times=(0.0,), V_rest=-65.0, tdelaybp=2e-3,
            tbss=25e-3, tbsf=3e-3, Ibsf=0.75, Ibss=0.25,
            maxBPAP=38.0, s_term=25.0, tep1=0.05, tep2=0.005,
            sum_pulses=False):

    t = sym.Symbol("t")
    v = sym.Float(V_rest)
    for t0 in times:
        te = t - t0                # time since EPSP onset (= release time)
        tb = t - t0 - tdelaybp     # time since bAP onset
        EPSP = sym.Piecewise(
            (0.0, te < 0),
            (s_term * (sym.exp(-te / tep1) - sym.exp(-te / tep2)), True))
        BPAP = sym.Piecewise(
            (0.0, tb < 0),
            (maxBPAP * (Ibsf * sym.exp(-tb / tbsf)
                        + Ibss * sym.exp(-tb / tbss)), True))
        pulse_v = EPSP + BPAP
        if sum_pulses:
            v += pulse_v
        else:
            v = sym.Piecewise((v, te < 0), (sym.Float(V_rest) + pulse_v, True))
    return v


def hold_voltage(V_mV=-65.0):
    return float(V_mV)


def voltage_onsets(times, tdelaybp=2e-3):
    onsets = []
    for t0 in times:
        onsets.append(float(t0))
        onsets.append(float(t0) + tdelaybp)
    return sorted(set(onsets))
