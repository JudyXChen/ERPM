"""IP3R -- DYK 8-state on ERm.  IPR_ijk: i=IP3, j=act Ca, k=inhib Ca.

Open state IPR_110 (IP3 bound, act Ca bound, inhib free).  12 reactions:
    a1/b1 (IP3 binding when k=0), a3/b3 (IP3 when k=1),
    a4/b4 (inhib Ca when i=0), a2/b2 (inhib when i=1),
    a5/b5 (act Ca, all faces).

Ca flux through IPR_110. 
**** Ca_ER side scaled by V_spine/V_ER.
"""

from smart.model_assembly import Parameter, Reaction, Species
from smart.units import unit


sec = unit.second
um = unit.micrometer
dimensionless = unit.dimensionless
D_unit = um**2 / sec


def register(p):

    erm = "ERm"

    species = [
        Species("IPR_000", 1.0, dimensionless, 0.0, D_unit, erm),
        Species("IPR_100", 0.0, dimensionless, 0.0, D_unit, erm),
        Species("IPR_010", 0.0, dimensionless, 0.0, D_unit, erm),
        Species("IPR_001", 0.0, dimensionless, 0.0, D_unit, erm),
        Species("IPR_110", 0.0, dimensionless, 0.0, D_unit, erm),
        Species("IPR_101", 0.0, dimensionless, 0.0, D_unit, erm),
        Species("IPR_011", 0.0, dimensionless, 0.0, D_unit, erm),
        Species("IPR_111", 0.0, dimensionless, 0.0, D_unit, erm),
    ]

    params = [
        Parameter("a1", p.a1, dimensionless),
        Parameter("b1", p.b1, dimensionless),
        Parameter("a2", p.a2, dimensionless),
        Parameter("b2", p.b2, dimensionless),
        Parameter("a3", p.a3, dimensionless),
        Parameter("b3", p.b3, dimensionless),
        Parameter("a4", p.a4, dimensionless),
        Parameter("b4", p.b4, dimensionless),
        Parameter("a5", p.a5, dimensionless),
        Parameter("b5", p.b5, dimensionless),
        Parameter("k_IP3R_flux", p.k_IP3R_flux, dimensionless),
    ]

    S_IP3R = float(p.N_IP3R) / (float(p.N_A) * float(p.V_spine_L))
    V_ratio = float(p.V_spine) / float(p.V_ER)

    def _rx(name, lhs, rhs, kf, kr, eqn_f, eqn_r, smap):
        fs = {}
        if "Ca_c" in smap:
            fs["Ca_c"] = S_IP3R
        if "IP3" in smap:
            fs["IP3"] = S_IP3R
        if "Ca_ER" in smap:
            fs["Ca_ER"] = V_ratio * S_IP3R
        return Reaction(
            name, lhs, rhs,
            param_map={"kf": kf, "kr": kr},
            eqn_f_str=eqn_f,
            eqn_r_str=eqn_r,
            species_map=smap,
            flux_scaling=fs,
            explicit_restriction_to_domain=erm,
        )

    reactions = [
        # IP3 binding (k=0)
        _rx("IP3R_J1", ["IPR_000", "IP3"], ["IPR_100"],
            "a1", "b1", "kf*IP3*IPR_000", "kr*IPR_100",
            {"IPR_000": "IPR_000", "IPR_100": "IPR_100", "IP3": "IP3"}),
        _rx("IP3R_J5", ["IPR_010", "IP3"], ["IPR_110"],
            "a1", "b1", "kf*IP3*IPR_010", "kr*IPR_110",
            {"IPR_010": "IPR_010", "IPR_110": "IPR_110", "IP3": "IP3"}),
        # IP3 binding (k=1)
        _rx("IP3R_J3", ["IPR_001", "IP3"], ["IPR_101"],
            "a3", "b3", "kf*IP3*IPR_001", "kr*IPR_101",
            {"IPR_001": "IPR_001", "IPR_101": "IPR_101", "IP3": "IP3"}),
        _rx("IP3R_J7", ["IPR_011", "IP3"], ["IPR_111"],
            "a3", "b3", "kf*IP3*IPR_011", "kr*IPR_111",
            {"IPR_011": "IPR_011", "IPR_111": "IPR_111", "IP3": "IP3"}),
        # Inhibitory Ca binding (i=0)
        _rx("IP3R_J4", ["IPR_000"], ["IPR_001"],
            "a4", "b4", "kf*Ca_c*IPR_000", "kr*IPR_001",
            {"IPR_000": "IPR_000", "IPR_001": "IPR_001", "Ca_c": "Ca_c"}),
        _rx("IP3R_J8", ["IPR_010"], ["IPR_011"],
            "a4", "b4", "kf*Ca_c*IPR_010", "kr*IPR_011",
            {"IPR_010": "IPR_010", "IPR_011": "IPR_011", "Ca_c": "Ca_c"}),
        # Inhibitory Ca binding (i=1)
        _rx("IP3R_J2", ["IPR_100"], ["IPR_101"],
            "a2", "b2", "kf*Ca_c*IPR_100", "kr*IPR_101",
            {"IPR_100": "IPR_100", "IPR_101": "IPR_101", "Ca_c": "Ca_c"}),
        _rx("IP3R_J6", ["IPR_110"], ["IPR_111"],
            "a2", "b2", "kf*Ca_c*IPR_110", "kr*IPR_111",
            {"IPR_110": "IPR_110", "IPR_111": "IPR_111", "Ca_c": "Ca_c"}),
        # Activating Ca binding
        _rx("IP3R_J9", ["IPR_000"], ["IPR_010"],
            "a5", "b5", "kf*Ca_c*IPR_000", "kr*IPR_010",
            {"IPR_000": "IPR_000", "IPR_010": "IPR_010", "Ca_c": "Ca_c"}),
        _rx("IP3R_J10", ["IPR_100"], ["IPR_110"],
            "a5", "b5", "kf*Ca_c*IPR_100", "kr*IPR_110",
            {"IPR_100": "IPR_100", "IPR_110": "IPR_110", "Ca_c": "Ca_c"}),
        _rx("IP3R_J11", ["IPR_001"], ["IPR_011"],
            "a5", "b5", "kf*Ca_c*IPR_001", "kr*IPR_011",
            {"IPR_001": "IPR_001", "IPR_011": "IPR_011", "Ca_c": "Ca_c"}),
        _rx("IP3R_J12", ["IPR_101"], ["IPR_111"],
            "a5", "b5", "kf*Ca_c*IPR_101", "kr*IPR_111",
            {"IPR_101": "IPR_101", "IPR_111": "IPR_111", "Ca_c": "Ca_c"}),

        Reaction(
            "IP3R_flux",
            ["Ca_ER"], ["Ca_c"],
            param_map={"k": "k_IP3R_flux"},
            eqn_f_str="k*IPR_110*(Ca_ER - Ca_c)",
            species_map={
                "IPR_110": "IPR_110", "Ca_c": "Ca_c", "Ca_ER": "Ca_ER",
            },
            flux_scaling={
                "Ca_c": float(p.N_IP3R) / (float(p.N_A) * float(p.V_spine_L)),
                "Ca_ER": (float(p.V_spine) / float(p.V_ER))
                * float(p.N_IP3R) / (float(p.N_A) * float(p.V_spine_L)),
            },
            explicit_restriction_to_domain=erm,
        ),
    ]

    return species, params, reactions
