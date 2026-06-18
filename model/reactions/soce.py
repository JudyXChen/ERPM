from smart.model_assembly import Parameter, Reaction, Species
from smart.units import unit


sec = unit.second
um = unit.micrometer
dimensionless = unit.dimensionless
D_unit = um**2 / sec


def register(p):
    species = [
        Species("P_O_STIM1", 0.0, dimensionless, 0.0, D_unit, "ERm"),
    ]

    params = [
        Parameter("K_STIM1", p.K_STIM1, dimensionless),
        Parameter("tau_STIM1", p.tau_STIM1, dimensionless),
        Parameter("w_STIM1", p.w_STIM1, dimensionless),
        Parameter("N_Orai1", p.N_Orai1, dimensionless),
        Parameter("g_Orai1", p.g_Orai1, dimensionless),
    ]
    erm = "ERm"

    reactions = [
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
        Reaction(
            "SOCE_Ca_entry", [], ["Ca_c"],
            param_map={
                "N_Orai1": "N_Orai1",
                "g_Orai1": "g_Orai1",
                "w_STIM1": "w_STIM1",
                "F_const": "F_const",
                "V_spine_L": "V_spine_L",
                "RT_F": "RT_F",
                "Ca_ext": "Ca_ext",
                "V_m": "V_m",
            },

            eqn_f_str=(
                "N_Orai1*w_STIM1*P_O_STIM1*g_Orai1"
                "*(0.5*RT_F*ln(Ca_ext/(Ca_c + 1e-12)) - V_m*1e-3)"
                "/(2.0*F_const*V_spine_L)"
            ),
            species_map={
                "P_O_STIM1": "P_O_STIM1",
                "Ca_c": "Ca_c",
            },
            explicit_restriction_to_domain=erm,
        ),
    ]

    return species, params, reactions
