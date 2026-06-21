"""Parameter groups and sampling ranges for the global sensitivity analysis.

Sampling is multiplicative / log-uniform: each parameter is drawn over
[nominal / factor, nominal * factor] in log10 space.  FACTORS sets the spread
per group (geometry is tighter because the MCell mesh constrains it; rates and
counts are the genuinely uncertain knobs).  Per-parameter overrides go in
RANGE_OVERRIDES.

Edit GROUPS to re-cut the grouping; everything downstream keys off it.
"""

# group -> parameters it owns.  Order is cosmetic; names must exist in
# model.parameters.default_parameters().
GROUPS = {
    "geometry": ["V_spine", "V_ER"],
    "serca_uptake": [
        "N_SERCA", "k_x0x1", "k_x1x0", "k_x1x2", "k_x2x1", "k_x2y2",
        "k_y2x2", "k_y2y1", "k_y1y2", "k_y1y0", "k_y0y1", "k_y0x0", "k_x0y0",
    ],
    "serca_leak": ["k_S_leak"],
    "ip3r": [
        "N_IP3R", "k_IP3R_flux",
        "a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4", "a5", "b5",
    ],
    "ryr": [
        "N_RyR", "k_RyR", "k_AB", "k_AU", "k_IC", "k_IO", "k_IU",
        "alpha_RyR", "beta_RyR", "delta_prime", "delta_RyR", "gamma_RyR",
    ],
    "pmca_ncx": [
        "N_PMCA", "N_NCX", "k_P1", "k_P2", "k_P3", "k_P_leak",
        "k_NCX1", "k_NCX2", "k_NCX3", "k_NCX_leak",
    ],
    "buffers": [
        "Bm_total", "Bf_total", "k_Bm_on", "k_Bm_off",
        "k_Bf_on", "k_Bf_off", "k_d",
    ],
    "ip3_basal": ["IP3_basal", "k_f", "k_deg"],
    "setpoints": ["Ca_ER_init", "Ca_c_init", "Ca_ext"],
    # SOCE is only meaningful with include_soce=True; excluded by default.
    "soce": ["N_Orai1", "g_Orai1", "K_STIM1", "tau_STIM1", "w_STIM1"],
}

# Default multiplicative spread per group (sampled log-uniform over [1/f, f]).
FACTORS = {
    "geometry": 2.0,        # MCell mesh pins this; vary only modestly
    "setpoints": 3.0,       # Ca_ER/Ca_c/Ca_ext are fairly well known
    "_default": 10.0,       # rates and channel counts: an order of magnitude
}

# Per-parameter absolute (low, high) overrides; bypasses the factor rule.
RANGE_OVERRIDES = {}

# Groups that require include_soce=True to have any effect.
SOCE_GROUPS = {"soce"}


def active_groups(include_soce=False):
    """The grouping dict actually used for a run."""
    return {
        g: ps for g, ps in GROUPS.items()
        if include_soce or g not in SOCE_GROUPS
    }


def factor_for(group):
    return FACTORS.get(group, FACTORS["_default"])
