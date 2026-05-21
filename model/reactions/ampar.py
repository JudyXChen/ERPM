"""AMPAR -- 7-state Jonas on PM. Glu buffer only (Ca-impermeable).

No Ca flux. Glu mass balance: forward of A0->A1, A1->A2, A3->A4 each consume one Glu.
"""

from smart.model_assembly import Parameter, Reaction, Species
from smart.units import unit


sec = unit.second
um = unit.micrometer
dimensionless = unit.dimensionless
D_unit = um**2 / sec


def register(p):

    species = [
        Species("A0", 1.0, dimensionless, 0.0, D_unit, "PM"),
        Species("A1", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("A2", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("AO", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("A3", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("A4", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("A5", 0.0, dimensionless, 0.0, D_unit, "PM"),
    ]

    params = [
        Parameter("k_A_C0C1", p.k_A_C0C1, dimensionless),
        Parameter("k_A_C1C0", p.k_A_C1C0, dimensionless),
        Parameter("k_A_C1C2", p.k_A_C1C2, dimensionless),
        Parameter("k_A_C2C1", p.k_A_C2C1, dimensionless),
        Parameter("k_A_C20", p.k_A_C20, dimensionless),
        Parameter("k_A_C02", p.k_A_C02, dimensionless),
        Parameter("k_A_C1C3", p.k_A_C1C3, dimensionless),
        Parameter("k_A_C3C1", p.k_A_C3C1, dimensionless),
        Parameter("k_A_C3C4", p.k_A_C3C4, dimensionless),
        Parameter("k_A_C4C3", p.k_A_C4C3, dimensionless),
        Parameter("k_A_C2C4", p.k_A_C2C4, dimensionless),
        Parameter("k_A_C4C2", p.k_A_C4C2, dimensionless),
        Parameter("k_A_C4C5", p.k_A_C4C5, dimensionless),
        Parameter("k_A_C5C4", p.k_A_C5C4, dimensionless),
        Parameter("k_A_C50", p.k_A_C50, dimensionless),
        Parameter("k_A_0C5", p.k_A_0C5, dimensionless),
    ]

    pm = "PM"
    reactions = [
        Reaction(
            "A_A0_A1", ["A0", "Glu"], ["A1"],
            param_map={"kf": "k_A_C0C1", "kr": "k_A_C1C0"},
            eqn_f_str="kf*Glu*A0",
            eqn_r_str="kr*A1",
            species_map={"A0": "A0", "A1": "A1", "Glu": "Glu"},
            explicit_restriction_to_domain=pm,
        ),
        Reaction(
            "A_A1_A2", ["A1", "Glu"], ["A2"],
            param_map={"kf": "k_A_C1C2", "kr": "k_A_C2C1"},
            eqn_f_str="kf*Glu*A1",
            eqn_r_str="kr*A2",
            species_map={"A1": "A1", "A2": "A2", "Glu": "Glu"},
            explicit_restriction_to_domain=pm,
        ),
        Reaction(
            "A_A2_AO", ["A2"], ["AO"],
            param_map={"kf": "k_A_C20", "kr": "k_A_C02"},
            eqn_f_str="kf*A2",
            eqn_r_str="kr*AO",
            species_map={"A2": "A2", "AO": "AO"},
            explicit_restriction_to_domain=pm,
        ),
        Reaction(
            "A_A1_A3", ["A1"], ["A3"],
            param_map={"kf": "k_A_C1C3", "kr": "k_A_C3C1"},
            eqn_f_str="kf*A1",
            eqn_r_str="kr*A3",
            species_map={"A1": "A1", "A3": "A3"},
            explicit_restriction_to_domain=pm,
        ),
        Reaction(
            "A_A3_A4", ["A3", "Glu"], ["A4"],
            param_map={"kf": "k_A_C3C4", "kr": "k_A_C4C3"},
            eqn_f_str="kf*Glu*A3",
            eqn_r_str="kr*A4",
            species_map={"A3": "A3", "A4": "A4", "Glu": "Glu"},
            explicit_restriction_to_domain=pm,
        ),
        Reaction(
            "A_A2_A4", ["A2"], ["A4"],
            param_map={"kf": "k_A_C2C4", "kr": "k_A_C4C2"},
            eqn_f_str="kf*A2",
            eqn_r_str="kr*A4",
            species_map={"A2": "A2", "A4": "A4"},
            explicit_restriction_to_domain=pm,
        ),
        Reaction(
            "A_A4_A5", ["A4"], ["A5"],
            param_map={"kf": "k_A_C4C5", "kr": "k_A_C5C4"},
            eqn_f_str="kf*A4",
            eqn_r_str="kr*A5",
            species_map={"A4": "A4", "A5": "A5"},
            explicit_restriction_to_domain=pm,
        ),
        Reaction(
            "A_A5_AO", ["A5"], ["AO"],
            param_map={"kf": "k_A_C50", "kr": "k_A_0C5"},
            eqn_f_str="kf*A5",
            eqn_r_str="kr*AO",
            species_map={"A5": "A5", "AO": "AO"},
            explicit_restriction_to_domain=pm,
        ),
    ]
    
    S_AMPAR = float(p.N_AMPAR) / (float(p.N_A) * float(p.V_spine_L))
    for r in reactions:
        if "Glu" in (r.species_map or {}):
            r.flux_scaling = {"Glu": S_AMPAR}

    return species, params, reactions
