import sympy as sym


# Glutamate release size used by HM / ODE_MODEL_PLAN.md.
GLUT_RELEASE_MOLECULES = 500.0


def glu_bolus_from_molecules(n_molecules, p):
    """Concentration (M) for ``n_molecules`` released into the spine volume. """
    return float(n_molecules) / (p.N_A * p.V_spine_L)


def resolve_glu(glu, p):
    """Map a ``glu`` argument to a concentration in M."""
    if glu is None:
        return 0.0
    if isinstance(glu, str):
        if glu == "glut_release":
            return glu_bolus_from_molecules(GLUT_RELEASE_MOLECULES, p)
        raise ValueError(f"unknown glu protocol {glu!r}")
    return float(glu)


def hold_voltage(V_mV=-70.0):
    return float(V_mV)


def bap_voltage_expr(
    t_stim=0.0,
    V_rest=-65.0,
    tdelaybp=2e-3,
    tbss=25e-3,
    tbsf=3e-3,
    Ibsf=0.75,
    Ibss=0.25,
    maxBPAP=38.0,
    s_term=25,
    tdelay=0.0,
    tep1=0.05,
    tep2=0.005,
    causal=False,
):
    t = sym.Symbol("t")
    te = t - t_stim - tdelay      # time since EPSP onset
    tb = t - t_stim - tdelaybp    # time since bAP onset

    EPSP_raw = s_term * (sym.exp(-te / tep1) - sym.exp(-te / tep2))
    BPAP_raw = maxBPAP * (
        Ibsf * sym.exp(-tb / tbsf) + Ibss * sym.exp(-tb / tbss)
    )
    EPSP = sym.Piecewise((0.0, te < 0), (EPSP_raw, True))
    BPAP = (
        sym.Piecewise((0.0, tb < 0), (BPAP_raw, True))
        if causal
        else BPAP_raw
    )
    return float(V_rest) + EPSP + BPAP


def apply(p, *, glu="glut_release", voltage=-70.0):
    """Set the glutamate release on ``p`` and return the voltage expression."""
    p.Glu_init = resolve_glu(glu, p)
    return voltage
