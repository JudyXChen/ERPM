"""Model regression tests: rest calibration, flux balance, RyR gating, Gq-PLC pathway."""
import unittest

import numpy as np
from scipy.integrate import solve_ivp

from model import stimulus
from model.ode0d import _make_voltage, _solve_with_pulses, build_system, equilibrate_state, rhs
from model.parameters import balanced_r15_scale, default_parameters
from model.reactions import mglur_gq_plc
from scripts.er_flux_decomp import flux_series
from scripts.ryr_conductance import build_generator, stationary_distribution
from scripts.run_conditions import pulse_table, time_grid


def net(item, state, voltage=-65.0):
    _name, forward, f_order, reverse, r_order, _stoich = item
    value = forward(*(state[i] for i in f_order), voltage) if forward else 0.0
    if reverse:
        value -= reverse(*(state[i] for i in r_order), voltage)
    return float(value)


class RestCalibrationTests(unittest.TestCase):
    def test_default_rest_hits_targets(self):
        p = default_parameters()
        p.Glu_init = 0.0
        names, y0, compiled, _ = build_system(p)
        eq = equilibrate_state(y0, compiled, _make_voltage(-65.0), names=names)
        state = dict(zip(names, eq.state))
        self.assertTrue(eq.converged)
        self.assertAlmostEqual(state["Ca_c"] / 100e-9, 1.0, places=5)
        self.assertAlmostEqual(state["Ca_ER"] / 150e-6, 1.0, places=5)
        self.assertAlmostEqual(state["IP3"] / 50e-9, 1.0, places=5)

    def test_r15_scale_follows_resting_ip3_and_decay(self):
        for ip3, k16 in ((20e-9, 0.28), (50e-9, 0.125), (100e-9, 1.0)):
            p = default_parameters()
            p.Glu_init = 0.0
            p.IP3_basal, p.plc_gq_k16 = ip3, k16
            p.plc_gq_r15_scale = balanced_r15_scale(p)
            names, y0, compiled, _ = build_system(p)
            eq = equilibrate_state(y0, compiled, _make_voltage(-65.0), names=names, max_time_s=4000)
            self.assertAlmostEqual(eq.state[names.index("IP3")] / ip3, 1.0, places=4)

    def test_plc_binding_cycle_is_closed(self):
        p = default_parameters()
        forward = (p.plc_gq_kf9 / p.plc_gq_kr9) * (p.plc_gq_kf10 / p.plc_gq_kr10)
        reverse = (p.plc_gq_kf8 / p.plc_gq_kr8) * (p.plc_gq_kf11 / p.plc_gq_kr11)
        self.assertAlmostEqual(forward / reverse, 1.0)

    def test_widely_spaced_pulses_have_finite_voltage(self):
        voltage = _make_voltage(stimulus.voltage(times=[0.0, 5.0, 10.0], V_rest=-65.0))
        values = np.array([voltage(t) for t in np.linspace(0.0, 11.0, 2001)])
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(voltage(4.99), -65.0, delta=1.0)


class ERRepairTests(unittest.TestCase):
    def test_unsampled_solver_boundary_preserves_elapsed_time(self):
        def decay(t,y): return -y
        t=np.array([0.,.1,1.])
        sol=_solve_with_pulses(decay,(0,1),np.array([1.]),method='BDF',t_eval=t,
                               args=(),extra_boundaries=[.35],rtol=1e-10,atol=1e-13)
        np.testing.assert_allclose(sol.y[0],np.exp(-t),rtol=1e-8)

    def test_resting_balance_and_small_soce_baseline_shift(self):
        rests=[]
        for soce in (False,True):
            # A low threshold keeps SOCE near-silent, so the baseline shift stays small.
            p=default_parameters();p.Glu_init=0;p.K_STIM1=37.5e-6
            names,y,c,_=build_system(p,include_soce=soce)
            eq=equilibrate_state(y,c,_make_voltage(-65),names=names)
            self.assertTrue(eq.converged)
            rests.append(eq.state[names.index('Ca_ER')])
            # Perturb Ca to exercise channel binding contributions to the budget.
            probe=eq.state.copy();probe[names.index('Ca_c')]=1e-6
            flux=flux_series(p,names,c,np.array([0.]),probe[:,None],_make_voltage(-65))
            deriv=rhs(0,probe,c,_make_voltage(-65))
            for name,comp,vol in [('Ca_c','cytosol',p.V_spine_L),('Ca_ER','ER',p.V_ER*1e-15)]:
                exact=deriv[names.index(name)]*p.N_A*vol
                self.assertAlmostEqual(exact,flux[comp]['net'][0],places=6)
        self.assertAlmostEqual(rests[0]*1e6,150,delta=.01)
        self.assertLess(abs(rests[1]/rests[0]-1),.01)


class StimulusProtocolTests(unittest.TestCase):
    def test_single_pulse_keeps_original_grid(self):
        expected = np.unique(np.r_[np.linspace(0, .1, 10001), np.linspace(.1, 10, 4951),
                                   np.linspace(10, 60, 2501), np.linspace(60, 300, 2401)])
        np.testing.assert_allclose(time_grid([0.0]), expected, atol=1e-9)

    def test_train_grid_samples_each_release_finely(self):
        pulses = stimulus.release_times(0.2, 6)
        self.assertEqual(pulses, [0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
        grid = time_grid(pulses)
        self.assertEqual(grid[-1], 300.0)
        for start in pulses:
            self.assertIn(start, grid)
            self.assertAlmostEqual(np.diff(grid[(grid >= start) & (grid <= start + .1)]).max(),
                                   4e-5, places=9)

    def test_pulse_table_finds_minimum_between_releases(self):
        t = np.array([0., 1., 5., 6., 10.])
        rows = pulse_table(t, np.array([150., 40., 70., 30., 90.]), [0., 5.])
        self.assertEqual([r['ER_min_uM'] for r in rows], [40., 30.])
        self.assertEqual(rows[1]['ER_before_uM'], 70.)
        self.assertEqual(rows[1]['ER_min_after_pulse_s'], 1.)


class RyRActivationGateTests(unittest.TestCase):
    def test_inactivated_opening_requires_four_sites(self):
        names, generator, _, _ = build_generator(default_parameters())
        index = {name: i for i, name in enumerate(names)}
        q = generator(1e-7, 150e-6)
        for activation in range(5):
            for inhibition in range(5):
                ci = index[f'R_CI_{activation}_{inhibition}']
                oi = index[f'R_OI_{activation}_{inhibition}']
                self.assertEqual(q[oi, ci], 650 if activation == 4 else 0)
                self.assertEqual(q[ci, oi], 60)
        np.testing.assert_allclose(q.sum(axis=0), 0, atol=1e-12)
        occupancy = stationary_distribution(q)
        opened = sum(occupancy[i] for i, name in enumerate(names)
                     if name.startswith('R_O_'))
        self.assertLess(opened, 2e-5)
        self.assertGreater(opened, 0)

    def test_ten_micromolar_pulse_activates_and_recovers(self):
        from scipy.linalg import expm
        names, generator, _, _ = build_generator(default_parameters())
        opened=[i for i,name in enumerate(names) if name.startswith("R_O_")]
        qrest=generator(.073e-6,150e-6)
        rest=stationary_distribution(qrest)
        activated=expm(generator(10e-6,150e-6)*.1)@rest
        recovered=expm(qrest*.9)@activated
        self.assertLess(rest[opened].sum(),1e-5)
        self.assertGreater(activated[opened].sum(),.01)
        self.assertLess(recovered[opened].sum(),activated[opened].sum()/100)
        self.assertAlmostEqual(recovered.sum(),1,places=9)


class MglurGqPlcTests(unittest.TestCase):
    def setUp(self):
        self.p = default_parameters()
        self.names, self.y, compiled, _ = build_system(
            self.p, ryr_off=True, plc="new")
        self.index = {name: i for i, name in enumerate(self.names)}
        self.reactions = {item[0]: item for item in compiled}

    def test_new_is_default_and_old_remains_selectable(self):
        self.p.k_S_leak = 0.0
        self.p.k_S_leak_gq = 0.2
        default = build_system(self.p, ryr_off=True)
        explicit = build_system(self.p, ryr_off=True, plc="new")
        self.assertEqual(default[0], explicit[0])
        self.assertEqual([r[0] for r in default[2]],
                         [r[0] for r in explicit[2]])
        old = build_system(self.p, ryr_off=True, plc="old")
        self.assertIn("Gq_GDP", default[0])
        self.assertIn("PLC_free", old[0])
        self.assertNotIn("Gq_GDP", old[0])
        old_leak = {r[0]: r for r in old[2]}["S_leak"]
        new_leak = {r[0]: r for r in default[2]}["S_leak"]
        self.assertEqual(net(old_leak, old[1]), 0.0)
        self.assertAlmostEqual(net(new_leak, default[1]),
                               0.2 * self.p.Ca_ER_init)
        with self.assertRaises(ValueError):
            build_system(self.p, plc="hybrid")

    def test_receptor_binding_has_no_parallel_catalytic_unbinding(self):
        values = {"Glu": 2e-6, "mGluR_free": 0.4,
                  "mGluR_Glu": 0.3, "mGluR_Glu2": 0.3}
        for name, value in values.items():
            self.y[self.index[name]] = value
        expected = {
            "mG_bind1": self.p.k_mG_f * 0.4 * 2e-6 - self.p.k_mG_b * 0.3,
            "mG_bind2": self.p.k_GG_f * 0.3 * 2e-6 - self.p.k_GG_b * 0.3,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(net(self.reactions[name], self.y), value)
        _species, _params, reactions = mglur_gq_plc.register(self.p)
        reaction_names = {r.name for r in reactions}
        self.assertNotIn("mG_cat1", reaction_names)
        self.assertNotIn("mG_cat2", reaction_names)
        sources = {r.name for r in reactions
                   if "IP3" in r.lhs or "IP3" in r.rhs}
        self.assertEqual(sources, {"R14", "R15", "R16_IP3_decay"})

    def test_effective_activation_rate_and_zero_ligand(self):
        self.y[self.index["mGluR_Glu"]] = 0.2
        self.y[self.index["mGluR_Glu2"]] = 0.3
        self.y[self.index["Gq_GDP"]] = 8000.0
        expected = (self.p.plc_gq_k_act * 0.5 * 8000.0
                    / self.p.mGluR_total)
        self.assertAlmostEqual(net(self.reactions["Gq_activation"], self.y),
                               expected)
        self.y[self.index["mGluR_Glu"]] = 0.0
        self.y[self.index["mGluR_Glu2"]] = 0.0
        self.assertEqual(net(self.reactions["Gq_activation"], self.y), 0.0)

    def test_r7_r16_rates_and_surface_scaling(self):
        values = {"IP3": 3e-8, "Ca_c": 3e-7, "Gq_GDP": 9.0,
                  "Gq_GTP": 10.0, "PLC": 11.0, "PLC_GqGTP": 12.0,
                  "PLC_Ca": 13.0, "PLC_Ca_GqGTP": 14.0}
        for name, value in values.items():
            self.y[self.index[name]] = value
        p = self.p
        expected = {
            "R7": p.plc_gq_kf7 * 10,
            "R8": p.plc_gq_kf8 * 10 * 11 - p.plc_gq_kr8 * 12,
            "R9": p.plc_gq_kf9 * 3e-7 * 11 - p.plc_gq_kr9 * 13,
            "R10": p.plc_gq_kf10 * 10 * 13 - p.plc_gq_kr10 * 14,
            "R11": p.plc_gq_kf11 * 3e-7 * 12 - p.plc_gq_kr11 * 14,
            "R12": p.plc_gq_kf12 * 14,
            "R13": p.plc_gq_kf13 * 12,
            "R14": p.plc_gq_kcat14 * 14 * p.plc_gq_pip2_density
                   / (p.plc_gq_km14 + p.plc_gq_pip2_density),
            "R15": p.plc_gq_r15_scale * p.plc_gq_kcat15 * 13
                   * p.plc_gq_pip2_density
                   / (p.plc_gq_km15 + p.plc_gq_pip2_density),
            "R16_IP3_decay": p.plc_gq_k16 * 3e-8,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(net(self.reactions[name], self.y), value,
                                   delta=max(abs(value), 1.0) * 1e-12)
        scale = p.A_PM / (p.N_A * p.V_spine_L)
        ip3 = self.index["IP3"]
        self.assertAlmostEqual(self.reactions["R14"][-1][ip3], scale)
        self.assertAlmostEqual(self.reactions["R15"][-1][ip3], scale)

    def test_receptor_gq_and_plc_conservation(self):
        pools = (
            {"mGluR_free", "mGluR_Glu", "mGluR_Glu2"},
            {"Gq_GDP", "Gq_GTP", "PLC_GqGTP", "PLC_Ca_GqGTP"},
            {"PLC", "PLC_GqGTP", "PLC_Ca", "PLC_Ca_GqGTP"},
        )
        pathway = {"mG_bind1", "mG_bind2", "Gq_activation",
                   *(f"R{i}" for i in range(7, 14))}
        for name in pathway:
            stoich = self.reactions[name][-1]
            for pool in pools:
                self.assertAlmostEqual(sum(
                    stoich.get(self.index[species], 0.0) for species in pool),
                    0.0, msg=name)
        state = dict(zip(self.names, self.y))
        self.assertAlmostEqual(state["Gq_GDP"], self.p.plc_gq_total_density)
        self.assertAlmostEqual(sum(state[n] for n in pools[2]),
                               self.p.N_PLC / self.p.A_PM)

    def test_explicit_old_trajectory_is_unchanged(self):
        p = default_parameters()
        # Legacy reference values predate the rest and VSCC calibration.
        p.k_P_leak, p.k_NCX_leak = 4.3, 19.4
        p.N_VSCC, p.k_AU, p.IP3_basal, p.k_mG_b = 1.0, 60.0, 2.0e-8, 7.85
        p.Glu_init = 500.0 / (p.N_A * p.V_spine_L)
        names, y0, compiled, _ = build_system(p, ryr_off=True, plc="old")
        solution = solve_ivp(
            rhs, (0.0, 0.5), y0, method="BDF", t_eval=[0.5],
            args=(compiled, _make_voltage(-65.0)), rtol=1e-8, atol=1e-14)
        expected = {"Glu": 2.1801689493948632e-09,
                    "IP3": 2.3719133177121963e-08,
                    "Ca_c": 7.04907808154694e-08,
                    "Ca_ER": 1.55515727582972e-04}
        index = {name: i for i, name in enumerate(names)}
        for name, value in expected.items():
            self.assertAlmostEqual(solution.y[index[name], -1], value,
                                   delta=1e-13)

    def test_ryr_on_assembly_uses_existing_open_states(self):
        names, _y0, compiled, _ = build_system(self.p, plc="new")
        self.assertIn("R_O_4_4", names)
        self.assertNotIn("R_O_5_0", names)
        self.assertIn("RyR_CICR", {reaction[0] for reaction in compiled})


if __name__ == "__main__":
    unittest.main()
