from smart.model_assembly import Parameter, Reaction, Species
from smart.units import unit


sec = unit.second
um = unit.micrometer
dimensionless = unit.dimensionless
D_unit = um**2 / sec


def register(p):
    species = [
        Species("Glu", float(getattr(p, "Glu_init", 0.0)),
                dimensionless, 0.0, D_unit, "Cyto"),
        Species("IP3", float(getattr(p, "IP3_basal", 0.0)),
                dimensionless, 0.0, D_unit, "Cyto"),
        Species("Ca_c", float(getattr(p, "Ca_c_init", 1e-7)),
                dimensionless, 0.0, D_unit, "Cyto"),
        Species("Ca_ER", float(getattr(p, "Ca_ER_init", 5e-4)),
                dimensionless, 0.0, D_unit, "ER"),
        # mGluR on PM 
        Species("mGluR_free", float(getattr(p, "mGluR_total", 0.0)),
                dimensionless, 0.0, D_unit, "PM"),
        Species("mGluR_Glu", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("mGluR_Glu2", 0.0, dimensionless, 0.0, D_unit, "PM"),
        # PLC in Cyto
        Species("PLC_free", float(getattr(p, "PLC_total", 0.0)),
                dimensionless, 0.0, D_unit, "Cyto"),
        Species("PLC_Ca", 0.0, dimensionless, 0.0, D_unit, "Cyto"),
        Species("PLC_Ca_PIP2", 0.0, dimensionless, 0.0, D_unit, "Cyto"),
    ]

    params = [
        Parameter("k_mG_f", p.k_mG_f, dimensionless),
        Parameter("k_mG_b", p.k_mG_b, dimensionless),
        Parameter("k_GG_f", p.k_GG_f, dimensionless),
        Parameter("k_GG_b", p.k_GG_b, dimensionless),
        Parameter("k_IP", p.k_IP, dimensionless),
        Parameter("k_IP_2", p.k_IP_2, dimensionless),
        Parameter("k_PLC_fCa", p.k_PLC_fCa, dimensionless),
        Parameter("k_PLC_bCa", p.k_PLC_bCa, dimensionless),
        Parameter("k_PLC_fP", p.k_PLC_fP, dimensionless),
        Parameter("k_PLC_bP", p.k_PLC_bP, dimensionless),
        Parameter("k_f", p.k_f, dimensionless),
        Parameter("k_deg", p.k_deg, dimensionless),
        Parameter("IP3_basal", p.IP3_basal, dimensionless),
    ]

    pm = "PM"

    reactions = [
        # mGluR_free + Glu  <->  mGluR_Glu  (PM membrane receptor binding).
        Reaction(
            "mG_bind1", ["mGluR_free", "Glu"], ["mGluR_Glu"],
            param_map={"kf": "k_mG_f", "kr": "k_mG_b"},
            eqn_f_str="kf*mGluR_free*Glu",
            eqn_r_str="kr*mGluR_Glu",
            species_map={
                "mGluR_free": "mGluR_free", "mGluR_Glu": "mGluR_Glu",
                "Glu": "Glu",
            },
            explicit_restriction_to_domain=pm,
        ),
        # Catalytic IP3 production from singly-bound mGluR. 
        Reaction(
            "mG_cat1", ["mGluR_Glu"], ["mGluR_free", "Glu", "IP3"],
            param_map={"k": "k_IP"},
            eqn_f_str="k*mGluR_Glu",
            species_map={
                "mGluR_free": "mGluR_free", "mGluR_Glu": "mGluR_Glu",
                "Glu": "Glu", "IP3": "IP3",
            },
            explicit_restriction_to_domain=pm,
        ),
        # mGluR_Glu + Glu  <->  mGluR_Glu2.
        Reaction(
            "mG_bind2", ["mGluR_Glu", "Glu"], ["mGluR_Glu2"],
            param_map={"kf": "k_GG_f", "kr": "k_GG_b"},
            eqn_f_str="kf*mGluR_Glu*Glu",
            eqn_r_str="kr*mGluR_Glu2",
            species_map={
                "mGluR_Glu": "mGluR_Glu", "mGluR_Glu2": "mGluR_Glu2",
                "Glu": "Glu",
            },
            explicit_restriction_to_domain=pm,
        ),
        # Catalytic IP3 production from doubly-bound mGluR. 
        Reaction(
            "mG_cat2", ["mGluR_Glu2"], ["mGluR_Glu", "Glu", "IP3"],
            param_map={"k": "k_IP_2"},
            eqn_f_str="k*mGluR_Glu2",
            species_map={
                "mGluR_Glu": "mGluR_Glu", "mGluR_Glu2": "mGluR_Glu2",
                "Glu": "Glu", "IP3": "IP3",
            },
            explicit_restriction_to_domain=pm,
        ),
        # PLC_free + Ca_c  <->  PLC_Ca  (cytosolic).
        Reaction(
            "PLC_bindCa", ["PLC_free", "Ca_c"], ["PLC_Ca"],
            param_map={"kf": "k_PLC_fCa", "kr": "k_PLC_bCa"},
            eqn_f_str="kf*PLC_free*Ca_c",
            eqn_r_str="kr*PLC_Ca",
            species_map={
                "PLC_free": "PLC_free", "PLC_Ca": "PLC_Ca", "Ca_c": "Ca_c",
            },
        ),
        # PLC_Ca <-> PLC_Ca_PIP2 (PIP2 absorbed into rate constant).
        Reaction(
            "PLC_bindPIP2", ["PLC_Ca"], ["PLC_Ca_PIP2"],
            param_map={"kf": "k_PLC_fP", "kr": "k_PLC_bP"},
            eqn_f_str="kf*PLC_Ca",
            eqn_r_str="kr*PLC_Ca_PIP2",
            species_map={
                "PLC_Ca": "PLC_Ca", "PLC_Ca_PIP2": "PLC_Ca_PIP2",
            },
        ),
        # PLC_Ca_PIP2 -> PLC_Ca + IP3.
        Reaction(
            "PLC_cleave", ["PLC_Ca_PIP2"], ["PLC_Ca", "IP3"],
            param_map={"k": "k_f"},
            eqn_f_str="k*PLC_Ca_PIP2",
            species_map={
                "PLC_Ca": "PLC_Ca", "PLC_Ca_PIP2": "PLC_Ca_PIP2", "IP3": "IP3",
            },
        ),

        Reaction(
            "IP3_decay", ["IP3"], [],
            param_map={"k": "k_deg"},
            eqn_f_str="k*IP3",
            species_map={"IP3": "IP3"},
        ),
    ]

    S_mGluR = float(p.N_mGluR) / (float(p.N_A) * float(p.V_spine_L))
    S_PLC = float(p.N_PLC) / (float(p.N_A) * float(p.V_spine_L))
    _bulk = {"Glu", "IP3", "Ca_c"}
    for r in reactions:
        sm = r.species_map or {}
        if any(s.startswith("mGluR") for s in sm):
            r.flux_scaling = {b: S_mGluR for b in sm if b in _bulk}
        elif any(s.startswith("PLC") for s in sm):
            sc = {b: S_PLC for b in sm if b in _bulk}
            if sc:
                r.flux_scaling = sc

    return species, params, reactions
