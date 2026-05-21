from smart.model_assembly import Parameter, Reaction, Species
from smart.units import unit


sec = unit.second
um = unit.micrometer
dimensionless = unit.dimensionless
D_unit = um**2 / sec


def register(p):
    species = [
        Species("PMCA", 1.0, dimensionless, 0.0, D_unit, "PM"),
        Species("PMCA1", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("NCX", 1.0, dimensionless, 0.0, D_unit, "PM"),
        Species("NCX1", 0.0, dimensionless, 0.0, D_unit, "PM"),
    ]

    params = [
        Parameter("k_P1", p.k_P1, dimensionless),
        Parameter("k_P2", p.k_P2, dimensionless),
        Parameter("k_P3", p.k_P3, dimensionless),
        Parameter("k_P_leak", p.k_P_leak, dimensionless),
        Parameter("k_NCX1", p.k_NCX1, dimensionless),
        Parameter("k_NCX2", p.k_NCX2, dimensionless),
        Parameter("k_NCX3", p.k_NCX3, dimensionless),
        Parameter("k_NCX_leak", p.k_NCX_leak, dimensionless),
    ]

    pm = "PM"

    S_PMCA = float(p.N_PMCA) / (float(p.N_A) * float(p.V_spine_L))
    S_NCX = float(p.N_NCX) / (float(p.N_A) * float(p.V_spine_L))

    reactions = [
        # PMCA: Ca + PMCA <-> PMCA1.
        Reaction(
            "P_bind", ["PMCA", "Ca_c"], ["PMCA1"],
            param_map={"kf": "k_P1", "kr": "k_P2"},
            eqn_f_str="kf*Ca_c*PMCA",
            eqn_r_str="kr*PMCA1",
            species_map={"PMCA": "PMCA", "PMCA1": "PMCA1", "Ca_c": "Ca_c"},
            flux_scaling={"Ca_c": S_PMCA},
            explicit_restriction_to_domain=pm,
        ),
        # PMCA1 -> PMCA: irreversible translocation, Ca leaves to extracellular.
        Reaction(
            "P_translocate", ["PMCA1"], ["PMCA"],
            param_map={"k": "k_P3"},
            eqn_f_str="k*PMCA1",
            species_map={"PMCA": "PMCA", "PMCA1": "PMCA1"},
            explicit_restriction_to_domain=pm,
        ),
        # PMCA passive leak. *** Require optimization ***
        Reaction(
            "P_leak", [], ["Ca_c"],
            param_map={"k": "k_P_leak"},
            eqn_f_str="k*PMCA",
            species_map={"PMCA": "PMCA", "Ca_c": "Ca_c"},
            flux_scaling={"Ca_c": S_PMCA},
            explicit_restriction_to_domain=pm,
        ),
        # NCX reactions
        Reaction(
            "N_bind", ["NCX", "Ca_c"], ["NCX1"],
            param_map={"kf": "k_NCX1", "kr": "k_NCX2"},
            eqn_f_str="kf*Ca_c*NCX",
            eqn_r_str="kr*NCX1",
            species_map={"NCX": "NCX", "NCX1": "NCX1", "Ca_c": "Ca_c"},
            flux_scaling={"Ca_c": S_NCX},
            explicit_restriction_to_domain=pm,
        ),
        Reaction(
            "N_translocate", ["NCX1"], ["NCX"],
            param_map={"k": "k_NCX3"},
            eqn_f_str="k*NCX1",
            species_map={"NCX": "NCX", "NCX1": "NCX1"},
            explicit_restriction_to_domain=pm,
        ),
        # leak *** Require optimization ***
        Reaction(
            "N_leak", [], ["Ca_c"],
            param_map={"k": "k_NCX_leak"},
            eqn_f_str="k*NCX",
            species_map={"NCX": "NCX", "Ca_c": "Ca_c"},
            flux_scaling={"Ca_c": S_NCX},
            explicit_restriction_to_domain=pm,
        ),
    ]

    return species, params, reactions
