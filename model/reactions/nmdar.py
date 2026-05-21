from smart.model_assembly import Parameter, Reaction, Species
from smart.units import unit


sec = unit.second
um = unit.micrometer
dimensionless = unit.dimensionless
D_unit = um**2 / sec


def register(p):
    species = [
        Species("R0", 1.0, dimensionless, 0.0, D_unit, "PM"),
        Species("R1", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("R2", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("RD", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("R", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("R0B", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("R1B", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("R2B", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("RDB", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("RB", 0.0, dimensionless, 0.0, D_unit, "PM"),
    ]

    params = [
        Parameter("k_N_C0C1", p.k_N_C0C1, dimensionless),
        Parameter("k_N_C1C0", p.k_N_C1C0, dimensionless),
        Parameter("k_N_C1C2", p.k_N_C1C2, dimensionless),
        Parameter("k_N_C2C1", p.k_N_C2C1, dimensionless),
        Parameter("k_C2D", p.k_C2D, dimensionless),
        Parameter("k_DC2", p.k_DC2, dimensionless),
        Parameter("k_N_C20", p.k_N_C20, dimensionless),
        Parameter("k_N_0C2", p.k_N_0C2, dimensionless),
        Parameter("k_C20b", p.k_C20b, dimensionless),
        Parameter("k_0C2b", p.k_0C2b, dimensionless),
        Parameter("gamma_N", p.gamma_N, dimensionless),
        Parameter("V_r_N", p.V_r_N, dimensionless),
    ]

    pm = "PM"
    reactions = [
        # Unblocked branch: Glu binding chain.
        Reaction(
            "N_R0_R1", ["R0", "Glu"], ["R1"],
            param_map={"kf": "k_N_C0C1", "kr": "k_N_C1C0"},
            eqn_f_str="kf*Glu*R0",
            eqn_r_str="kr*R1",
            species_map={"R0": "R0", "R1": "R1", "Glu": "Glu"},
            explicit_restriction_to_domain=pm,
        ),
        Reaction(
            "N_R1_R2", ["R1", "Glu"], ["R2"],
            param_map={"kf": "k_N_C1C2", "kr": "k_N_C2C1"},
            eqn_f_str="kf*Glu*R1",
            eqn_r_str="kr*R2",
            species_map={"R1": "R1", "R2": "R2", "Glu": "Glu"},
            explicit_restriction_to_domain=pm,
        ),
        # R2 <-> RD desensitisation (no Glu).
        Reaction(
            "N_R2_RD", ["R2"], ["RD"],
            param_map={"kf": "k_C2D", "kr": "k_DC2"},
            eqn_f_str="kf*R2",
            eqn_r_str="kr*RD",
            species_map={"R2": "R2", "RD": "RD"},
            explicit_restriction_to_domain=pm,
        ),
        # R2 <-> R opening.
        Reaction(
            "N_R2_R", ["R2"], ["R"],
            param_map={"kf": "k_N_C20", "kr": "k_N_0C2"},
            eqn_f_str="kf*R2",
            eqn_r_str="kr*R",
            species_map={"R2": "R2", "R": "R"},
            explicit_restriction_to_domain=pm,
        ),
        # Voltage-dependent Mg block of the open state.
        Reaction(
            "N_R_RB", ["R"], ["RB"],
            param_map={"V_m": "V_m"},
            eqn_f_str="1200.0*exp(-V_m/17.0)*R",
            eqn_r_str="10800.0*exp(V_m/47.0)*RB",
            species_map={"R": "R", "RB": "RB"},
            explicit_restriction_to_domain=pm,
        ),
        # Blocked branch mirrors unblocked Glu / desens chain.
        Reaction(
            "N_R0B_R1B", ["R0B", "Glu"], ["R1B"],
            param_map={"kf": "k_N_C0C1", "kr": "k_N_C1C0"},
            eqn_f_str="kf*Glu*R0B",
            eqn_r_str="kr*R1B",
            species_map={"R0B": "R0B", "R1B": "R1B", "Glu": "Glu"},
            explicit_restriction_to_domain=pm,
        ),
        Reaction(
            "N_R1B_R2B", ["R1B", "Glu"], ["R2B"],
            param_map={"kf": "k_N_C1C2", "kr": "k_N_C2C1"},
            eqn_f_str="kf*Glu*R1B",
            eqn_r_str="kr*R2B",
            species_map={"R1B": "R1B", "R2B": "R2B", "Glu": "Glu"},
            explicit_restriction_to_domain=pm,
        ),
        Reaction(
            "N_R2B_RDB", ["R2B"], ["RDB"],
            param_map={"kf": "k_C2D", "kr": "k_DC2"},
            eqn_f_str="kf*R2B",
            eqn_r_str="kr*RDB",
            species_map={"R2B": "R2B", "RDB": "RDB"},
            explicit_restriction_to_domain=pm,
        ),
        Reaction(
            "N_R2B_RB", ["R2B"], ["RB"],
            param_map={"kf": "k_C20b", "kr": "k_0C2b"},
            eqn_f_str="kf*R2B",
            eqn_r_str="kr*RB",
            species_map={"R2B": "R2B", "RB": "RB"},
            explicit_restriction_to_domain=pm,
        ),
        # Ca2+ entry through the open state. V_m AND V_r_N are both in mV
        # (Bell 2022 convention, matching the Mg-block 17/47 mV constants
        # and VSCC gating Vc constants -- the rest of the model uses V_m
        # in mV too). gamma_N = 4.5e-15 (Bell 2022 Table 2) thus carries
        # units of A/mV, so gamma_N*(V_m-V_r_N) is in A and dividing by
        # 2*q_e gives Ca2+ ions/s per channel x open fraction R. At
        # V_m=-27 mV (bAP peak) the result matches HM's MCell
        # nmdar_rate_scaled within 2x. Bulk d[Ca_c]/dt comes from
        # flux_scaling = N_NMDAR/(N_A*V_spine_L) -- same convention as
        # Bf_total. PREVIOUS BUG (fixed 2026-05-20): had V_m*1e-3 here
        # which silently divided the per-channel Ca flux by 1000.
        Reaction(
            "N_Ca_entry", [], ["Ca_c"],
            param_map={
                "gamma_N": "gamma_N",
                "V_r_N": "V_r_N",
                "q_e": "q_e",
                "V_m": "V_m",
            },
            eqn_f_str="-(gamma_N*(V_m - V_r_N)/(2.0*q_e))*R",
            species_map={"R": "R", "Ca_c": "Ca_c"},
            flux_scaling={
                "Ca_c": float(p.N_NMDAR)
                / (float(p.N_A) * float(p.V_spine_L))
            },
            explicit_restriction_to_domain=pm,
        ),
    ]

    # NMDAR states are occupancy fractions; the Glu consumed by the
    # Glu-binding steps must be scaled to the per-spine NMDAR population
    # (S_NMDAR = N_NMDAR/(N_A*V_spine_L)) -- otherwise ~6 receptors drain
    # the bulk Glu pool unphysically. Markov transitions stay unscaled.
    # See doc/flux_units_plan.md.
    S_NMDAR = float(p.N_NMDAR) / (float(p.N_A) * float(p.V_spine_L))
    for r in reactions:
        if "Glu" in (r.species_map or {}):
            r.flux_scaling = {"Glu": S_NMDAR}

    return species, params, reactions
