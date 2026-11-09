"""Default parameter set for the well-mixed ODE model.

Concentrations in M, rates in s^-1 or (M*s)^-1, voltages in mV.
"""

from types import SimpleNamespace


# Physical constants
N_A = 6.02214076e23
F_const = 96485.33212
q_e = 1.602176634e-19
R_gas = 8.31446261815324
T_kelvin = 310.0
RT_F = R_gas * T_kelvin / F_const

# Spine geometry computed directly from the MCell ideal_homog mesh
V_spine_um3 = 1.05 
V_ER_um3 = 0.05249
A_PM_um2 = 7.179
A_PSD_um2 = 0.21
A_ER_um2 = 1.284
V_spine_L = V_spine_um3 * 1e-15
V_ER_L = V_ER_um3 * 1e-15


def balanced_r15_scale(p):
    """R15 scale (s15) that balances R16 decay at the resting IP3 (IP3_basal).

    k16*IP3_basal = s15*kcat15*PLC_Ca*PIP2/(Km15+PIP2)*A_PM/(N_A*V_spine_L)
    """
    pip2 = p.plc_gq_pip2_density
    plc_ca = (p.N_PLC / p.A_PM) * p.plc_gq_plc_ca_fraction
    production = (p.plc_gq_kcat15 * plc_ca * pip2 / (p.plc_gq_km15 + pip2)
                  * p.A_PM / (p.N_A * p.V_spine_L))
    return p.plc_gq_k16 * p.IP3_basal / production


def default_parameters():
    Bf_total = 4791.0 / (N_A * V_spine_L)

    p = SimpleNamespace(
        # Physical constants
        N_A=N_A,
        F_const=F_const,
        q_e=q_e,
        RT_F=RT_F,
        # Geometry
        V_spine=V_spine_um3,
        V_ER=V_ER_um3,
        V_spine_L=V_spine_L,
        A_PM=A_PM_um2,
        # Initial concentrations / extracellular (M)
        Ca_c_init=1.0e-7,
        Ca_ER_init=150e-6,
        Ca_ext=2.0e-3,
        IP3_basal=5.0e-8,
        Bm_total=2.0e-5,
        Bf_total=Bf_total,
        mGluR_total=1.0,
        PLC_total=1.0,
        glu_clearance_rate=200.0,
        Glu_init=0.0,
        N_NMDAR=35.0,
        N_VSCC=7.0,  # Calibrated to Benedetti 2025 Fig 7B; 2/um2 density would give 14.
        N_PMCA=7165.0,
        N_NCX=1019.0,
        N_SERCA=1284.0,
        N_RyR=60.0,
        N_IP3R=15.0,
        N_AMPAR=278.0,
        N_mGluR=4197.0,
        N_PLC=839.0,

        # NMDAR
        k_N_C0C1=2.0e7,
        k_N_C1C0=11.0,
        k_N_C1C2=1.0e7,
        k_N_C2C1=22.0,
        k_C2D=16.8,
        k_DC2=3.6,
        k_N_C20=93.0,
        k_N_0C2=183.2,
        k_C20b=97.0,
        k_0C2b=574.0,
        gamma_N=4.5e-15,
        V_r_N=3.0,

        # AMPAR
        k_A_C0C1=9.18e6,
        k_A_C1C0=8520.0,
        k_A_C1C2=5.68e7,
        k_A_C2C1=6520.0,
        k_A_C20=8480.0,
        k_A_C02=1800.0,
        k_A_C1C3=5780.0,
        k_A_C3C1=78.4,
        k_A_C3C4=2.54e6,
        k_A_C4C3=91.4,
        k_A_C2C4=344.0,
        k_A_C4C2=1.45,
        k_A_C4C5=33.6,
        k_A_C5C4=380.8,
        k_A_C50=8.0,
        k_A_0C5=35.4,

        # VSCC 
        alpha_V_1=8080.0, beta_V_1=5760.0, V_const_1=49.14,
        alpha_V_2=13400.0, beta_V_2=12600.0, V_const_2=42.08,
        alpha_V_3=8780.0, beta_V_3=16300.0, V_const_3=55.31,
        alpha_V_4=34700.0, beta_V_4=3680.0, V_const_4=26.55,
        gamma_VSCC=3.72e-12,

        # PMCA / NCX (Table 5, Bartol 2015)
        k_P1=1.5e8, k_P2=15.0, k_P3=12.0,
        k_P_leak=6.832288382768589,  # scripts/calibrate_rest.py; Bartol's 4.3 is per total pump.
        k_NCX1=3.0e8, k_NCX2=300.0, k_NCX3=600.0,
        k_NCX_leak=20.496865148305766,  # scripts/calibrate_rest.py; Bartol's 19.4 is per total pump.

        # SOCE
        K_STIM1=60e-6,  # About 0.42 * rest, the ratio in Luik 2008 (169 uM at 400 uM ER).
        tau_STIM1=0.01,  # tau_SOCE, Francis 2026 (muscle), assumes STIM1-Orai1 already assembled at junctions.
        w_STIM1=1.0,
        N_Orai1=10.0,
        g_Orai1=4.752e-14 * A_PM_um2 / 10.0,  # S per channel: Francis 2026 density (S/um2) * A_PM / N_Orai1.

        # SERCA (Table 6, Bartol 2015)
        # Bartol 2015 Table 1 (Higgins 2006 at s = 1), with k_y2x2 = 0.97 (Higgins; Bartol lists 0.097) and two factors,
        # fitted at ER Ca = 0: s = 33.86 for turnover 5 cycles/s, g = 4.74 for K1/2 0.27 uM (Lytton 1992).
        # Off-rates / g: 83.7, 167.4.
        k_x0x1=2.0e8, k_x1x0=17.66,
        k_x1x2=1.0e8, k_x2x1=35.32,
        # 0.6 s, 0.97 s.
        k_x2y2=20.31, k_y2x2=32.84,
        k_y2y1=60.04, k_y1y2=1.0e5,
        k_y1y0=30.02, k_y0y1=2.0e5,
        # 0.4 s, 1.2e-3 s.
        k_y0x0=13.54, k_x0y0=0.04063,
        # Leak rate to balance the pump at rest. Not used in Gq-PLC version of model.
        k_S_leak=0, # 0.1873,
        # Separate leak for the Gq-PLC pathway.
        k_S_leak_gq=0.3349166725925168,  # scripts/calibrate_rest.py: Ca_c 100 nM, ER 150 uM.

        # RyR (Table 6, Tanskanen 2007 / Singh 2021)
        k_AB=3.0e7, k_AU=333.0,
        k_IC=4.5e5, k_IO=2.5e5, k_IU=1.0,
        alpha_RyR=650.0, beta_RyR=60.0,
        delta_prime=600.0, delta_RyR=75.0, gamma_RyR=5.0,
        k_RyR=1.09e9,

        # IP3R (Table 6, De Young-Keizer 1992 / Singh 2021)
        a1=4.0e8, b1=52.0,
        a2=2.0e5, b2=0.21,
        a3=4.0e8, b3=377.36,
        a4=2.0e5, b4=0.0289,
        a5=2.0e7, b5=1.6468,
        k_IP3R_flux=1.19e8,

        # Cytosolic buffers (Table 7, Bell 2022)
        k_Bm_on=1.0e6, k_Bm_off=1.0,
        k_Bf_on=1.0e6, k_Bf_off=2.0,
        k_d=50.0, # Optimization to derive

        # mGluR / PLC / IP3 cascade (Table 7, Singh 2021)
        k_mG_f=1.0e8, k_mG_b=78.5,  # Fitted to IP3 timing, x10 of Singh 2021 (7.85); also used by --plc old.
        k_GG_f=2.0e6, k_GG_b=15.7,
        k_IP=0.03875, k_IP_2=0.0775,
        k_PLC_fCa=3.0e6, k_PLC_bCa=1.0,
        k_PLC_fP=25.0, k_PLC_bP=10.0,
        k_f=0.155, k_deg=0.28,


        plc_gq_kf7=0.15,
        plc_gq_kf8=0.042, plc_gq_kr8=1.0,
        plc_gq_kf9=4.18e-3 * 1e6, plc_gq_kr9=0.0167,
        plc_gq_kf10=0.042, plc_gq_kr10=1.0,
        plc_gq_kf11=4.18e-3 * 1e6, plc_gq_kr11=0.0167,  # = R9: independent Ca/Gq sites close the binding cycle (Calizo: 3.34e4, 0.00334).
        plc_gq_kf12=6.0,
        plc_gq_kf13=6.0,
        plc_gq_kcat14=50.0, plc_gq_km14=653.0,  # kcat fitted to IP3 timing, x5 of Calizo (10).
       
        plc_gq_kcat15=0.333, plc_gq_km15=2587.0,
        plc_gq_pip2_density=4000.0,  # molecules/um2, fixed reservoir.
        plc_gq_total_density=10047.8611584221,
        plc_gq_k16=0.28,  # k_deg, Singh 2021 (Calizo 0.125).

        plc_gq_eta=1.0,
        plc_gq_k_act=1.291678304,  # s^-1; Calizo Kfr5 22.2 * (N_mGluR/A_PM) / Gq_total, not a fit.
        plc_gq_plc_ca_fraction=0.02441874050706858,
    )
    p.plc_gq_r15_scale = balanced_r15_scale(p)  # Keeps rest at IP3_basal.
    return p


DEFAULTS = default_parameters()
