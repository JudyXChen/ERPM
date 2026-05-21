"""VSCC -- 5-state Bartol gating + GHK Ca flux through V4 (open) on PM.

V0..V3 closed; V4 open. Forward rates exp(+V/V_const_i), reverse
exp(-V/V_const_i) (the Table-5 +V/+V form would make equilibrium V-
independent and is treated as a typo). GHK denominator guarded against
V=0 with a tiny offset baked into eqn_f_str.

Ca flux through V4 = gamma_VSCC * N_A / (2*F_const) * GHK(V) * V4.
"""

from smart.model_assembly import Parameter, Reaction, Species
from smart.units import unit


sec = unit.second
um = unit.micrometer
dimensionless = unit.dimensionless
D_unit = um**2 / sec


def register(p):
    """Build VSCC species, parameters, and reactions."""

    species = [
        Species("V0", 1.0, dimensionless, 0.0, D_unit, "PM"),
        Species("V1", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("V2", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("V3", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("V4", 0.0, dimensionless, 0.0, D_unit, "PM"),
    ]

    params = [
        Parameter("alpha_V_1", p.alpha_V_1, dimensionless),
        Parameter("beta_V_1", p.beta_V_1, dimensionless),
        Parameter("V_const_1", p.V_const_1, dimensionless),
        Parameter("alpha_V_2", p.alpha_V_2, dimensionless),
        Parameter("beta_V_2", p.beta_V_2, dimensionless),
        Parameter("V_const_2", p.V_const_2, dimensionless),
        Parameter("alpha_V_3", p.alpha_V_3, dimensionless),
        Parameter("beta_V_3", p.beta_V_3, dimensionless),
        Parameter("V_const_3", p.V_const_3, dimensionless),
        Parameter("alpha_V_4", p.alpha_V_4, dimensionless),
        Parameter("beta_V_4", p.beta_V_4, dimensionless),
        Parameter("V_const_4", p.V_const_4, dimensionless),
        Parameter("gamma_VSCC", p.gamma_VSCC, dimensionless),
    ]

    pm = "PM"

    # Each gating step is a single Reaction with V-dependent f/r rates.
    reactions = [
        Reaction(
            "V_V0_V1", ["V0"], ["V1"],
            param_map={
                "alpha": "alpha_V_1", "bta": "beta_V_1",
                "Vc": "V_const_1", "V_m": "V_m",
            },
            eqn_f_str="alpha*exp(V_m/Vc)*V0",
            eqn_r_str="bta*exp(-V_m/Vc)*V1",
            species_map={"V0": "V0", "V1": "V1"},
            explicit_restriction_to_domain=pm,
        ),
        Reaction(
            "V_V1_V2", ["V1"], ["V2"],
            param_map={
                "alpha": "alpha_V_2", "bta": "beta_V_2",
                "Vc": "V_const_2", "V_m": "V_m",
            },
            eqn_f_str="alpha*exp(V_m/Vc)*V1",
            eqn_r_str="bta*exp(-V_m/Vc)*V2",
            species_map={"V1": "V1", "V2": "V2"},
            explicit_restriction_to_domain=pm,
        ),
        Reaction(
            "V_V2_V3", ["V2"], ["V3"],
            param_map={
                "alpha": "alpha_V_3", "bta": "beta_V_3",
                "Vc": "V_const_3", "V_m": "V_m",
            },
            eqn_f_str="alpha*exp(V_m/Vc)*V2",
            eqn_r_str="bta*exp(-V_m/Vc)*V3",
            species_map={"V2": "V2", "V3": "V3"},
            explicit_restriction_to_domain=pm,
        ),
        Reaction(
            "V_V3_V4", ["V3"], ["V4"],
            param_map={
                "alpha": "alpha_V_4", "bta": "beta_V_4",
                "Vc": "V_const_4", "V_m": "V_m",
            },
            eqn_f_str="alpha*exp(V_m/Vc)*V3",
            eqn_r_str="bta*exp(-V_m/Vc)*V4",
            species_map={"V3": "V3", "V4": "V4"},
            explicit_restriction_to_domain=pm,
        ),
        # GHK Ca flux into Cyto. The +1e-9 offset on V_m guards the GHK
        # denominator at V=0; the small offset has negligible effect off zero.
        Reaction(
            "V_Ca_entry", [], ["Ca_c"],
            param_map={
                "gamma_VSCC": "gamma_VSCC",
                "N_A": "N_A",
                "F_const": "F_const",
                "V_m": "V_m",
            },
            eqn_f_str=(
                "gamma_VSCC*N_A/(2.0*F_const)"
                "*((0.393 - exp(-(V_m + 1e-9)/80.36))"
                "/(1.0 - exp(-(V_m + 1e-9)/80.36)))*V4"
            ),
            species_map={"V4": "V4", "Ca_c": "Ca_c"},
            # gamma_VSCC*N_A/(2F) = gamma/(2*q_e) = Ca ions/s per channel x
            # open fraction V4. Scale to the per-spine VSCC population /
            # cytosol volume: S_VSCC = N_VSCC/(N_A*V_spine_L).
            flux_scaling={
                "Ca_c": float(p.N_VSCC)
                / (float(p.N_A) * float(p.V_spine_L))
            },
            explicit_restriction_to_domain=pm,
        ),
    ]

    return species, params, reactions
