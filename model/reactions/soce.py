from smart.model_assembly import Parameter, Reaction, Species
from smart.units import unit


sec = unit.second
um = unit.micrometer
dimensionless = unit.dimensionless
D_unit = um**2 / sec


def register(p):
    species = [
        Species("P_O_STIM1", 0.0, dimensionless, 0.0, D_unit, "ERm"),
        Species("P_O_STIM2", 0.0, dimensionless, 0.0, D_unit, "ERm"),
    ]

    params = [
        Parameter("K_STIM1", p.K_STIM1, dimensionless),
        Parameter("K_STIM2", p.K_STIM2, dimensionless),
        Parameter("tau_STIM1", p.tau_STIM1, dimensionless),
        Parameter("tau_STIM2", p.tau_STIM2, dimensionless),
        Parameter("w_STIM1", p.w_STIM1, dimensionless),
        Parameter("w_STIM2", p.w_STIM2, dimensionless),
        Parameter("g_Orai1", p.g_Orai1, dimensionless),
    ]

    # ERm: adjacent to both ER (Ca_ER) and Cyto (Ca_c). See note above.
    erm = "ERm"

    reactions = [
        # P_O_STIM1 relaxes toward Hill4(Ca_ER; K_STIM1) with tau_STIM1.
        Reaction(
            "SOCE_PO_STIM1_relax", [], ["P_O_STIM1"],
            param_map={
                "tau": "tau_STIM1",
                "Kd": "K_STIM1",
            },
            eqn_f_str="(1.0/(1.0 + (Ca_ER/Kd)**4))/tau",
            eqn_r_str="P_O_STIM1/tau",
            species_map={"P_O_STIM1": "P_O_STIM1", "Ca_ER": "Ca_ER"},
            explicit_restriction_to_domain=erm,
        ),
        # P_O_STIM2 relaxes toward Hill4(Ca_ER; K_STIM2) with tau_STIM2.
        Reaction(
            "SOCE_PO_STIM2_relax", [], ["P_O_STIM2"],
            param_map={
                "tau": "tau_STIM2",
                "Kd": "K_STIM2",
            },
            eqn_f_str="(1.0/(1.0 + (Ca_ER/Kd)**4))/tau",
            eqn_r_str="P_O_STIM2/tau",
            species_map={"P_O_STIM2": "P_O_STIM2", "Ca_ER": "Ca_ER"},
            explicit_restriction_to_domain=erm,
        ),
        Reaction(
            "SOCE_Ca_entry", [], ["Ca_c"],
            param_map={
                "g_Orai1": "g_Orai1",
                "w_STIM1": "w_STIM1",
                "w_STIM2": "w_STIM2",
                "F_const": "F_const",
                "V_spine_L": "V_spine_L",
                "RT_F": "RT_F",
                "Ca_ext": "Ca_ext",
                "V_m": "V_m",
            },
            # UNITS NOTE: self-consistent in SI -- V_m converted from mV
            # to V (*1e-3) to match RT_F which is naturally in V (~0.0267).
            # The rest of the active model now uses V_m in mV everywhere
            # (nmdar.py, vscc.py). SOCE is currently disabled in
            # reactions/__init__.py; when re-enabled, this V-internal
            # convention works as-is. To unify to mV throughout, replace
            # `RT_F` -> `RT_F*1e3` and drop the `*1e-3` on V_m
            # (numerically equivalent).
            eqn_f_str=(
                "(w_STIM1*P_O_STIM1 + w_STIM2*P_O_STIM2)*g_Orai1"
                "*(0.5*RT_F*ln(Ca_ext/(Ca_c + 1e-12)) - V_m*1e-3)"
                "/(2.0*F_const*V_spine_L)"
            ),
            species_map={
                "P_O_STIM1": "P_O_STIM1",
                "P_O_STIM2": "P_O_STIM2",
                "Ca_c": "Ca_c",
            },
            explicit_restriction_to_domain=erm,
        ),
    ]

    return species, params, reactions
