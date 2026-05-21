from smart.model_assembly import Parameter, Reaction, Species
from smart.units import unit


sec = unit.second
um = unit.micrometer
dimensionless = unit.dimensionless
D_unit = um**2 / sec


def register(p):
    species = [
        Species("Sx", 1.0, dimensionless, 0.0, D_unit, "ERm"),
        Species("Sx1", 0.0, dimensionless, 0.0, D_unit, "ERm"),
        Species("Sx2", 0.0, dimensionless, 0.0, D_unit, "ERm"),
        Species("Sy2", 0.0, dimensionless, 0.0, D_unit, "ERm"),
        Species("Sy1", 0.0, dimensionless, 0.0, D_unit, "ERm"),
        Species("Sy", 0.0, dimensionless, 0.0, D_unit, "ERm"),
    ]

    params = [
        Parameter("k_x0x1", p.k_x0x1, dimensionless),
        Parameter("k_x1x0", p.k_x1x0, dimensionless),
        Parameter("k_x1x2", p.k_x1x2, dimensionless),
        Parameter("k_x2x1", p.k_x2x1, dimensionless),
        Parameter("k_x2y2", p.k_x2y2, dimensionless),
        Parameter("k_y2x2", p.k_y2x2, dimensionless),
        Parameter("k_y2y1", p.k_y2y1, dimensionless),
        Parameter("k_y1y2", p.k_y1y2, dimensionless),
        Parameter("k_y1y0", p.k_y1y0, dimensionless),
        Parameter("k_y0y1", p.k_y0y1, dimensionless),
        Parameter("k_y0x0", p.k_y0x0, dimensionless),
        Parameter("k_x0y0", p.k_x0y0, dimensionless),
        Parameter("k_S_leak", p.k_S_leak, dimensionless),
    ]

    erm = "ERm"

    S_SERCA = float(p.N_SERCA) / (float(p.N_A) * float(p.V_spine_L))
    V_ratio = float(p.V_spine) / float(p.V_ER)

    reactions = [
        # Cyto-side Ca binding
        Reaction(
            "S_x_x1", ["Sx", "Ca_c"], ["Sx1"],
            param_map={"kf": "k_x0x1", "kr": "k_x1x0"},
            eqn_f_str="kf*Ca_c*Sx",
            eqn_r_str="kr*Sx1",
            species_map={"Sx": "Sx", "Sx1": "Sx1", "Ca_c": "Ca_c"},
            flux_scaling={"Ca_c": S_SERCA},
            explicit_restriction_to_domain=erm,
        ),
        Reaction(
            "S_x1_x2", ["Sx1", "Ca_c"], ["Sx2"],
            param_map={"kf": "k_x1x2", "kr": "k_x2x1"},
            eqn_f_str="kf*Ca_c*Sx1",
            eqn_r_str="kr*Sx2",
            species_map={"Sx1": "Sx1", "Sx2": "Sx2", "Ca_c": "Ca_c"},
            flux_scaling={"Ca_c": S_SERCA},
            explicit_restriction_to_domain=erm,
        ),
        # Conformational flip x2 <-> y2: Ca side switches.
        Reaction(
            "S_x2_y2", ["Sx2"], ["Sy2"],
            param_map={"kf": "k_x2y2", "kr": "k_y2x2"},
            eqn_f_str="kf*Sx2",
            eqn_r_str="kr*Sy2",
            species_map={"Sx2": "Sx2", "Sy2": "Sy2"},
            explicit_restriction_to_domain=erm,
        ),
        # ER-side Ca release (each forward step releases 1 Ca to Ca_ER).
        Reaction(
            "S_y2_y1", ["Sy2"], ["Sy1", "Ca_ER"],
            param_map={"kf": "k_y2y1", "kr": "k_y1y2"},
            eqn_f_str="kf*Sy2",
            eqn_r_str="kr*Ca_ER*Sy1",
            species_map={"Sy2": "Sy2", "Sy1": "Sy1", "Ca_ER": "Ca_ER"},
            flux_scaling={"Ca_ER": V_ratio * S_SERCA},
            explicit_restriction_to_domain=erm,
        ),
        Reaction(
            "S_y1_y0", ["Sy1"], ["Sy", "Ca_ER"],
            param_map={"kf": "k_y1y0", "kr": "k_y0y1"},
            eqn_f_str="kf*Sy1",
            eqn_r_str="kr*Ca_ER*Sy",
            species_map={"Sy1": "Sy1", "Sy": "Sy", "Ca_ER": "Ca_ER"},
            flux_scaling={"Ca_ER": V_ratio * S_SERCA},
            explicit_restriction_to_domain=erm,
        ),
        # Reset y -> x.
        Reaction(
            "S_y_x", ["Sy"], ["Sx"],
            param_map={"kf": "k_y0x0", "kr": "k_x0y0"},
            eqn_f_str="kf*Sy",
            eqn_r_str="kr*Sx",
            species_map={"Sy": "Sy", "Sx": "Sx"},
            explicit_restriction_to_domain=erm,
        ),
        # ER -> cyto passive leak. *** Require optimization ***
        Reaction(
            "S_leak", ["Ca_ER"], ["Ca_c"],
            param_map={"k": "k_S_leak"},
            eqn_f_str="k*Ca_ER",
            species_map={"Ca_ER": "Ca_ER", "Ca_c": "Ca_c"},
            flux_scaling={"Ca_ER": float(p.V_spine) / float(p.V_ER)},
            explicit_restriction_to_domain=erm,
        ),
    ]

    return species, params, reactions
