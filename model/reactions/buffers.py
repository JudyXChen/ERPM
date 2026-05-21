"""Cyto Ca buffers -- mobile B_m, fixed B_f, lumped k_d decay (Bell).

**** OPEN ISSUE: Bell uses B_m+B_f OR k_d, not both simultaneously. 
"""

from smart.model_assembly import Parameter, Reaction, Species
from smart.units import unit


sec = unit.second
um = unit.micrometer
dimensionless = unit.dimensionless
D_unit = um**2 / sec


def register(p):

    species = [
        Species("Bm_free", float(getattr(p, "Bm_total", 0.0)),
                dimensionless, 0.0, D_unit, "Cyto"),
        Species("Bm_Ca", 0.0, dimensionless, 0.0, D_unit, "Cyto"),
        Species("Bf_free", float(getattr(p, "Bf_total", 0.0)),
                dimensionless, 0.0, D_unit, "Cyto"),
        Species("Bf_Ca", 0.0, dimensionless, 0.0, D_unit, "Cyto"),
    ]

    params = [
        Parameter("k_Bm_on", p.k_Bm_on, dimensionless),
        Parameter("k_Bm_off", p.k_Bm_off, dimensionless),
        Parameter("k_Bf_on", p.k_Bf_on, dimensionless),
        Parameter("k_Bf_off", p.k_Bf_off, dimensionless),
        Parameter("k_d", p.k_d, dimensionless),
    ]

    reactions = [
        # Mobile buffer Bm.
        Reaction(
            "Bm_bind", ["Bm_free", "Ca_c"], ["Bm_Ca"],
            param_map={"kf": "k_Bm_on", "kr": "k_Bm_off"},
            eqn_f_str="kf*Bm_free*Ca_c",
            eqn_r_str="kr*Bm_Ca",
            species_map={
                "Bm_free": "Bm_free", "Bm_Ca": "Bm_Ca", "Ca_c": "Ca_c",
            },
        ),
        # Fixed buffer Bf.
        Reaction(
            "Bf_bind", ["Bf_free", "Ca_c"], ["Bf_Ca"],
            param_map={"kf": "k_Bf_on", "kr": "k_Bf_off"},
            eqn_f_str="kf*Bf_free*Ca_c",
            eqn_r_str="kr*Bf_Ca",
            species_map={
                "Bf_free": "Bf_free", "Bf_Ca": "Bf_Ca", "Ca_c": "Ca_c",
            },
        ),
        # Lumped first-order decay of cyto Ca.
        Reaction(
            "Ca_lumped_decay", ["Ca_c"], [],
            param_map={"k": "k_d"},
            eqn_f_str="k*Ca_c",
            species_map={"Ca_c": "Ca_c"},
        ),
    ]

    return species, params, reactions
