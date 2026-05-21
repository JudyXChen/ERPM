from smart.model_assembly import Parameter, Reaction, Species
from smart.units import unit


sec = unit.second
um = unit.micrometer
dimensionless = unit.dimensionless
D_unit = um**2 / sec


_MACROS = ("C", "O", "CI", "OI")  # m = 0, 1, 2, 3


def _state_name(m, ja, ji):
    return f"R_{m}_{ja}_{ji}"


def register(p):
    erm = "ERm"

    # Species: R_m_ja_ji.  Initial population concentrated in C,0,0.
    species = []
    for m in _MACROS:
        for ja in range(5):
            for ji in range(5):
                init = 1.0 if (m == "C" and ja == 0 and ji == 0) else 0.0
                species.append(
                    Species(_state_name(m, ja, ji), init, dimensionless,
                            0.0, D_unit, erm)
                )

    params = [
        Parameter("k_AB", p.k_AB, dimensionless),
        Parameter("k_AU", p.k_AU, dimensionless),
        Parameter("k_IC", p.k_IC, dimensionless),
        Parameter("k_IO", p.k_IO, dimensionless),
        Parameter("k_IU", p.k_IU, dimensionless),
        Parameter("alpha_RyR", p.alpha_RyR, dimensionless),
        Parameter("beta_RyR", p.beta_RyR, dimensionless),
        Parameter("gamma_RyR", p.gamma_RyR, dimensionless),
        Parameter("delta_RyR", p.delta_RyR, dimensionless),
        Parameter("delta_prime", p.delta_prime, dimensionless),
        Parameter("k_RyR", p.k_RyR, dimensionless),
    ]

    reactions = []

    # RyR states are occupancy fractions (all 100 sum to 1). EVERY reaction
    # that couples a fraction to a real concentration -- the gating Ca_c
    # binding steps (activation/inactivation) AND the CICR flux -- must
    # convert the fraction to a per-spine open/binding-site concentration
    # via S_RyR = N_RyR/(N_A*V_spine_L) (same convention as Bf_total).
    # Applied as per-species flux_scaling on Ca_c/Ca_ER ONLY, so the Markov
    # transitions (the fractions) are unscaled. Physically this makes the
    # ~N_RyR gating sites deplete negligible bulk Ca, as they should.
    # See doc/flux_units_plan.md.
    S_RyR = float(p.N_RyR) / (float(p.N_A) * float(p.V_spine_L))
    V_ratio = float(p.V_spine) / float(p.V_ER)

    # Activation Ca binding: R_m_ja_ji + Ca_c  <->  R_m_(ja+1)_ji
    for m in _MACROS:
        for ja in range(4):
            for ji in range(5):
                lo = _state_name(m, ja, ji)
                hi = _state_name(m, ja + 1, ji)
                reactions.append(
                    Reaction(
                        f"RyR_act_{m}_{ja}{ji}",
                        [lo, "Ca_c"], [hi],
                        param_map={"k_AB": "k_AB", "k_AU": "k_AU"},
                        eqn_f_str=f"{4 - ja}*k_AB*Ca_c*{lo}",
                        eqn_r_str=f"{ja + 1}*k_AU*{hi}",
                        species_map={lo: lo, hi: hi, "Ca_c": "Ca_c"},
                        flux_scaling={"Ca_c": S_RyR},
                        explicit_restriction_to_domain=erm,
                    )
                )

    # Inactivation Ca binding: R_m_ja_ji + Ca_c  <->  R_m_ja_(ji+1)
    for m in _MACROS:
        k_on_name = "k_IC" if m in ("C", "CI") else "k_IO"
        for ja in range(5):
            for ji in range(4):
                lo = _state_name(m, ja, ji)
                hi = _state_name(m, ja, ji + 1)
                reactions.append(
                    Reaction(
                        f"RyR_inact_{m}_{ja}{ji}",
                        [lo, "Ca_c"], [hi],
                        param_map={"k_on": k_on_name, "k_IU": "k_IU"},
                        eqn_f_str=f"{4 - ji}*k_on*Ca_c*{lo}",
                        eqn_r_str=f"{ji + 1}*k_IU*{hi}",
                        species_map={lo: lo, hi: hi, "Ca_c": "Ca_c"},
                        flux_scaling={"Ca_c": S_RyR},
                        explicit_restriction_to_domain=erm,
                    )
                )

    # C <-> O gating.  Opening alpha only at ja=4.
    for ja in range(5):
        for ji in range(5):
            c = _state_name("C", ja, ji)
            o = _state_name("O", ja, ji)
            if ja == 4:
                reactions.append(
                    Reaction(
                        f"RyR_CO_{ja}{ji}",
                        [c], [o],
                        param_map={"alpha": "alpha_RyR", "bta": "beta_RyR"},
                        eqn_f_str=f"alpha*{c}",
                        eqn_r_str=f"bta*{o}",
                        species_map={c: c, o: o},
                        explicit_restriction_to_domain=erm,
                    )
                )
            else:
                # ja < 4: only closing direction.
                reactions.append(
                    Reaction(
                        f"RyR_OC_{ja}{ji}",
                        [o], [c],
                        param_map={"bta": "beta_RyR"},
                        eqn_f_str=f"bta*{o}",
                        species_map={c: c, o: o},
                        explicit_restriction_to_domain=erm,
                    )
                )

    # CI <-> OI gating: alpha forward, bta reverse, all (ja, ji).
    for ja in range(5):
        for ji in range(5):
            ci = _state_name("CI", ja, ji)
            oi = _state_name("OI", ja, ji)
            reactions.append(
                Reaction(
                    f"RyR_CIOI_{ja}{ji}",
                    [ci], [oi],
                    param_map={"alpha": "alpha_RyR", "bta": "beta_RyR"},
                    eqn_f_str=f"alpha*{ci}",
                    eqn_r_str=f"bta*{oi}",
                    species_map={ci: ci, oi: oi},
                    explicit_restriction_to_domain=erm,
                )
            )

    # C <-> CI inactivation. 
    for ja in range(5):
        c0 = _state_name("C", ja, 0)
        ci0 = _state_name("CI", ja, 0)
        reactions.append(
            Reaction(
                f"RyR_CICrec_{ja}0",
                [ci0], [c0],
                param_map={"gma": "gamma_RyR"},
                eqn_f_str=f"gma*{ci0}",
                species_map={c0: c0, ci0: ci0},
                explicit_restriction_to_domain=erm,
            )
        )
        for ji in range(1, 5):
            c = _state_name("C", ja, ji)
            ci = _state_name("CI", ja, ji)
            reactions.append(
                Reaction(
                    f"RyR_CCI_{ja}{ji}",
                    [c], [ci],
                    param_map={"d": "delta_prime", "gma": "gamma_RyR"},
                    eqn_f_str=f"d*{c}",
                    eqn_r_str=f"gma*{ci}",
                    species_map={c: c, ci: ci},
                    explicit_restriction_to_domain=erm,
                )
            )

    # O <-> OI inactivation. 
    for ja in range(5):
        o0 = _state_name("O", ja, 0)
        oi0 = _state_name("OI", ja, 0)
        reactions.append(
            Reaction(
                f"RyR_OIOrec_{ja}0",
                [oi0], [o0],
                param_map={"gma": "gamma_RyR"},
                eqn_f_str=f"gma*{oi0}",
                species_map={o0: o0, oi0: oi0},
                explicit_restriction_to_domain=erm,
            )
        )
        for ji in range(1, 5):
            o = _state_name("O", ja, ji)
            oi = _state_name("OI", ja, ji)
            reactions.append(
                Reaction(
                    f"RyR_OOI_{ja}{ji}",
                    [o], [oi],
                    param_map={"d": "delta_RyR", "gma": "gamma_RyR"},
                    eqn_f_str=f"d*{o}",
                    eqn_r_str=f"gma*{oi}",
                    species_map={o: o, oi: oi},
                    explicit_restriction_to_domain=erm,
                )
            )

    # CICR Ca flux: Ca_ER <-> Ca_c.  The O-state species are occupancy
    # *fractions* (all 100 RyR states sum to 1); k_RyR (M^-1 s^-1, Singh
    # 2021) expects an open-RyR *concentration*. Convert via
    # S_RyR = N_RyR/(N_A*V_spine_L) -- same convention as Bf_total. Applied
    # as per-species flux_scaling on Ca_c/Ca_ER only, so the gradient form
    # is unchanged and no Markov state is rescaled. See
    # doc/flux_units_plan.md.
    open_states = [_state_name("O", ja, ji) for ja in range(5) for ji in range(5)]
    open_sum = " + ".join(open_states)
    species_map = {n: n for n in open_states}
    species_map.update({"Ca_c": "Ca_c", "Ca_ER": "Ca_ER"})

    reactions.append(
        Reaction(
            "RyR_CICR",
            ["Ca_ER"], ["Ca_c"],
            param_map={"k_RyR": "k_RyR"},
            eqn_f_str=f"k_RyR*({open_sum})*(Ca_ER - Ca_c)",
            species_map=species_map,
            flux_scaling={"Ca_c": S_RyR, "Ca_ER": V_ratio * S_RyR},
            explicit_restriction_to_domain=erm,
        )
    )

    return species, params, reactions
