from smart.model_assembly import Parameter, Reaction, Species
from smart.units import unit


um = unit.micrometer
dimensionless = unit.dimensionless
D_unit = um**2 / unit.second


def _surface_rx(name, lhs, rhs, **kwargs):
    """Create one individual PM reaction without repeating its domain tag."""
    return Reaction(
        name, lhs, rhs, explicit_restriction_to_domain="PM", **kwargs)


def register(p):
    plc_total = float(p.N_PLC) / float(p.A_PM)
    plc_ca = plc_total * float(p.plc_gq_plc_ca_fraction)

    species = [
        Species("Glu", float(p.Glu_init), dimensionless, 0.0, D_unit, "Cyto"),
        Species("IP3", float(p.IP3_basal), dimensionless, 0.0, D_unit, "Cyto"),
        Species("Ca_c", float(p.Ca_c_init), dimensionless, 0.0, D_unit, "Cyto"),
        Species("Ca_ER", float(p.Ca_ER_init), dimensionless, 0.0, D_unit, "ER"),
        Species("mGluR_free", float(p.mGluR_total), dimensionless, 0.0, D_unit, "PM"),
        Species("mGluR_Glu", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("mGluR_Glu2", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("Gq_GDP", float(p.plc_gq_total_density), dimensionless, 0.0, D_unit, "PM"),
        Species("Gq_GTP", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("PLC", plc_total - plc_ca, dimensionless, 0.0, D_unit, "PM"),
        Species("PLC_GqGTP", 0.0, dimensionless, 0.0, D_unit, "PM"),
        Species("PLC_Ca", plc_ca, dimensionless, 0.0, D_unit, "PM"),
        Species("PLC_Ca_GqGTP", 0.0, dimensionless, 0.0, D_unit, "PM"),
    ]

    names = [
        "k_mG_f", "k_mG_b", "k_GG_f", "k_GG_b",
        "plc_gq_k_act", "plc_gq_eta", "plc_gq_receptor_total",
        *(f"plc_gq_kf{i}" for i in range(7, 14)),
        *(f"plc_gq_kr{i}" for i in range(7, 14)),
        "plc_gq_kcat14", "plc_gq_km14", "plc_gq_kcat15",
        "plc_gq_km15", "plc_gq_pip2_density", "plc_gq_k16",
        "plc_gq_r15_scale",
    ]
    values = {name: getattr(p, name) for name in names
              if name != "plc_gq_receptor_total"}
    values["plc_gq_receptor_total"] = max(float(p.mGluR_total), 1e-30)
    params = [Parameter(name, values[name], dimensionless) for name in names]

    reactions = [
        _surface_rx(
            "mG_bind1", ["mGluR_free", "Glu"], ["mGluR_Glu"],
            param_map={"kf": "k_mG_f", "kr": "k_mG_b"},
            eqn_f_str="kf*mGluR_free*Glu", eqn_r_str="kr*mGluR_Glu",
            species_map={"mGluR_free": "mGluR_free", "Glu": "Glu",
                         "mGluR_Glu": "mGluR_Glu"}),
        _surface_rx(
            "mG_bind2", ["mGluR_Glu", "Glu"], ["mGluR_Glu2"],
            param_map={"kf": "k_GG_f", "kr": "k_GG_b"},
            eqn_f_str="kf*mGluR_Glu*Glu", eqn_r_str="kr*mGluR_Glu2",
            species_map={"mGluR_Glu": "mGluR_Glu", "Glu": "Glu",
                         "mGluR_Glu2": "mGluR_Glu2"}),
        # Effective coupling
        _surface_rx(
            "Gq_activation", ["Gq_GDP"], ["Gq_GTP"],
            param_map={"k": "plc_gq_k_act", "eta": "plc_gq_eta",
                       "Rtot": "plc_gq_receptor_total"},
            eqn_f_str="k*(R1 + eta*R2)*GD/Rtot",
            species_map={"R1": "mGluR_Glu", "R2": "mGluR_Glu2",
                         "GD": "Gq_GDP"}),
        _surface_rx(
            "R7", ["Gq_GTP"], ["Gq_GDP"],
            param_map={"kf": "plc_gq_kf7", "kr": "plc_gq_kr7"},
            eqn_f_str="kf*GT", eqn_r_str="kr*GD",
            species_map={"GT": "Gq_GTP", "GD": "Gq_GDP"}),
        _surface_rx(
            "R8", ["Gq_GTP", "PLC"], ["PLC_GqGTP"],
            param_map={"kf": "plc_gq_kf8", "kr": "plc_gq_kr8"},
            eqn_f_str="kf*GT*PLC", eqn_r_str="kr*PLCGT",
            species_map={"GT": "Gq_GTP", "PLC": "PLC",
                         "PLCGT": "PLC_GqGTP"}),
        _surface_rx(
            "R9", ["Ca_c", "PLC"], ["PLC_Ca"],
            param_map={"kf": "plc_gq_kf9", "kr": "plc_gq_kr9"},
            eqn_f_str="kf*Ca*PLC", eqn_r_str="kr*PLCCa",
            species_map={"Ca": "Ca_c", "PLC": "PLC", "PLCCa": "PLC_Ca"}),
        _surface_rx(
            "R10", ["Gq_GTP", "PLC_Ca"], ["PLC_Ca_GqGTP"],
            param_map={"kf": "plc_gq_kf10", "kr": "plc_gq_kr10"},
            eqn_f_str="kf*GT*PLCCa", eqn_r_str="kr*PLCCaGT",
            species_map={"GT": "Gq_GTP", "PLCCa": "PLC_Ca",
                         "PLCCaGT": "PLC_Ca_GqGTP"}),
        _surface_rx(
            "R11", ["Ca_c", "PLC_GqGTP"], ["PLC_Ca_GqGTP"],
            param_map={"kf": "plc_gq_kf11", "kr": "plc_gq_kr11"},
            eqn_f_str="kf*Ca*PLCGT", eqn_r_str="kr*PLCCaGT",
            species_map={"Ca": "Ca_c", "PLCGT": "PLC_GqGTP",
                         "PLCCaGT": "PLC_Ca_GqGTP"}),
        _surface_rx(
            "R12", ["PLC_Ca_GqGTP"], ["Gq_GDP", "PLC_Ca"],
            param_map={"kf": "plc_gq_kf12", "kr": "plc_gq_kr12"},
            eqn_f_str="kf*PLCCaGT", eqn_r_str="kr*GD*PLCCa",
            species_map={"PLCCaGT": "PLC_Ca_GqGTP", "GD": "Gq_GDP",
                         "PLCCa": "PLC_Ca"}),
        _surface_rx(
            "R13", ["PLC_GqGTP"], ["Gq_GDP", "PLC"],
            param_map={"kf": "plc_gq_kf13", "kr": "plc_gq_kr13"},
            eqn_f_str="kf*PLCGT", eqn_r_str="kr*GD*PLC",
            species_map={"PLCGT": "PLC_GqGTP", "GD": "Gq_GDP",
                         "PLC": "PLC"}),
        _surface_rx(
            "R14", [], ["IP3"],
            param_map={"kcat": "plc_gq_kcat14", "km": "plc_gq_km14",
                       "pip2": "plc_gq_pip2_density"},
            eqn_f_str="kcat*PLCCaGT*pip2/(km + pip2)",
            species_map={"PLCCaGT": "PLC_Ca_GqGTP", "IP3": "IP3"}),
        _surface_rx(
            "R15", [], ["IP3"],
            param_map={"scale": "plc_gq_r15_scale", "kcat": "plc_gq_kcat15",
                       "km": "plc_gq_km15", "pip2": "plc_gq_pip2_density"},
            eqn_f_str="scale*kcat*PLCCa*pip2/(km + pip2)",
            species_map={"PLCCa": "PLC_Ca", "IP3": "IP3"}),
        Reaction(
            "R16_IP3_decay", ["IP3"], [],
            param_map={"k": "plc_gq_k16"}, eqn_f_str="k*IP3",
            species_map={"IP3": "IP3"}),
    ]

    bulk_scale = float(p.A_PM) / (float(p.N_A) * float(p.V_spine_L))
    receptor_scale = float(p.N_mGluR) / (float(p.N_A) * float(p.V_spine_L))
    for reaction in reactions:
        if reaction.name in {"mG_bind1", "mG_bind2"}:
            reaction.flux_scaling = {"Glu": receptor_scale}
        elif reaction.name in {"R9", "R11"}:
            reaction.flux_scaling = {"Ca_c": bulk_scale}
        elif reaction.name in {"R14", "R15"}:
            reaction.flux_scaling = {"IP3": bulk_scale}

    return species, params, reactions
