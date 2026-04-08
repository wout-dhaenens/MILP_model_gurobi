"""
test_electric_toy_milp.py
=========================
Comprehensive test suite for the Electrical PV-Battery MILP model
("Electric toy MILP different price scenarios_forked").

Test philosophy
---------------
Each test class targets a different layer of the model:
  1. Unit tests  – pure-Python physics helpers (no solver needed)
  2. Feasibility tests – solver must return "Optimal"
  3. Monotonicity tests – economic / physical intuition checks
  4. Edge-case tests – degenerate inputs that must not crash
  5. Constraint-integrity tests – verify every constraint is satisfied
     in the returned solution

Run with:
    python -m pytest test_electric_toy_milp.py -v
  or simply:
    python test_electric_toy_milp.py

NOTE: place this file in the same folder as
      "Electric toy MILP different price scenarios_forked.py"
"""

import math
import unittest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── import the model we want to test ─────────────────────────────────────────
# The source file name contains spaces; import via importlib so we don't need
# to rename it.
import importlib, sys
_mod = importlib.import_module(
    "Electric toy MILP different price scenarios_forked"
)

run_scenario_optimization = _mod.run_scenario_optimization
calculate_lorenz_cop      = _mod.calculate_lorenz_cop
load_price_data           = _mod.load_price_data
define_scenarios          = _mod.define_scenarios

CAPEX_BAT_ANNUAL  = _mod.CAPEX_BAT_ANNUAL
CAPEX_PV_ANNUAL   = _mod.CAPEX_PV_ANNUAL
ETA_BAT_CH        = _mod.ETA_BAT_CH
ETA_BAT_DIS       = _mod.ETA_BAT_DIS
P_BAT_POWER_RATIO = _mod.P_BAT_POWER_RATIO
P_LOAD            = _mod.P_LOAD
T                 = _mod.T
time              = _mod.time
dt                = _mod.dt
BIG_M             = _mod.BIG_M
I_SOLAR           = _mod.I_SOLAR        # pre-fetched/synthetic solar profile
COP_lorenz        = _mod.COP_lorenz     # pre-fetched/synthetic COP array

# Thermal constants are not present in this model; set stubs so the rest of
# the file can reference them without NameError.
eta_in = eta_out = 1.0
kWh_per_m = beta = gamma = loss_lids_kwh = None
temp_h = temp_c = None
P_THERMAL_LOAD = np.zeros(T)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared by multiple tests
# ─────────────────────────────────────────────────────────────────────────────
FLAT_PRICE = np.full(T, 0.20)   # €0.20 / kWh flat

# Peak-valley price: cheap at night / midday, expensive in morning & evening
PEAK_PRICE = np.array(
    [0.10, 0.10, 0.10, 0.10, 0.12, 0.15,
     0.40, 0.50, 0.40, 0.20, 0.15, 0.10,
     0.10, 0.10, 0.10, 0.15, 0.30, 0.50,
     0.60, 0.40, 0.20, 0.10, 0.10, 0.10]
)

ATOL = 1e-4   # absolute tolerance for constraint checks


def solve(price=None, name="test"):
    """Convenience wrapper; returns (opt_dict, res_dict) or raises.
    I_SOLAR and COP_lorenz are globals already set by the module at import time.
    """
    p = price if price is not None else FLAT_PRICE
    result = run_scenario_optimization(p, name)
    assert result is not None, f"Solver returned None for scenario '{name}'"
    return result   # (opt, res)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  UNIT TESTS  (no solver)
# ═════════════════════════════════════════════════════════════════════════════
class TestPhysicsHelpers(unittest.TestCase):

    def test_lorenz_cop_positive(self):
        """COP must always be > 1 for physically valid temperatures."""
        cop = calculate_lorenz_cop(temp_c, temp_h, 10.0)
        self.assertGreater(cop, 1.0)

    def test_lorenz_cop_increases_with_source_temp(self):
        """Higher source temperature → higher COP (easier to lift heat)."""
        cop_cold = calculate_lorenz_cop(temp_c, temp_h,  0.0)
        cop_warm = calculate_lorenz_cop(temp_c, temp_h, 15.0)
        self.assertGreater(cop_warm, cop_cold)

    def test_lorenz_cop_equal_temps(self):
        """Degenerate case: sink_in == sink_out – must not raise ZeroDivisionError."""
        cop = calculate_lorenz_cop(70.0, 70.0, 10.0)
        self.assertGreater(cop, 0.0)

    def test_kwh_per_m_positive(self):
        """Tank energy density must be positive."""
        self.assertGreater(kWh_per_m, 0.0)

    def test_beta_positive(self):
        """Wall-loss coefficient must be positive."""
        self.assertGreater(beta, 0.0)

    def test_gamma_positive(self):
        self.assertGreater(gamma, 0.0)

    def test_loss_lids_positive(self):
        self.assertGreater(loss_lids_kwh, 0.0)

    def test_loads_non_negative(self):
        self.assertTrue(np.all(P_LOAD >= 0))
        self.assertTrue(np.all(P_THERMAL_LOAD >= 0))


# ═════════════════════════════════════════════════════════════════════════════
# 2.  FEASIBILITY TESTS
# ═════════════════════════════════════════════════════════════════════════════
class TestFeasibility(unittest.TestCase):

    def test_base_scenario_optimal(self):
        """Standard inputs must yield an Optimal solution."""
        opt, _ = solve(name="feasibility_base")
        self.assertIsNotNone(opt)

    def test_zero_solar_feasible(self):
        """No PV available: system must still be feasible (grid only)."""
        opt, res = solve(solar=np.zeros(T), name="feasibility_no_pv")
        self.assertAlmostEqual(opt['C_PV'], 0.0, places=3)
        # All electricity from grid
        self.assertTrue(np.all(res['P_buy'] >= -ATOL))

    def test_zero_price_feasible(self):
        """Free electricity: solver should use as much HP / bat as needed."""
        opt, res = solve(price=np.zeros(T), name="feasibility_free_elec")
        self.assertIsNotNone(opt)

    def test_high_thermal_load_feasible(self):
        """3× thermal load – solver must still find a feasible solution."""
        opt, res = solve(
            solar=SOLAR_PROFILE,
            price=FLAT_PRICE,
            name="feasibility_high_thermal"
        )
        self.assertIsNotNone(opt)

    def test_peak_price_feasible(self):
        opt, _ = solve(price=PEAK_PRICE, name="feasibility_peak_price")
        self.assertIsNotNone(opt)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MONOTONICITY / ECONOMIC-INTUITION TESTS
# ═════════════════════════════════════════════════════════════════════════════
class TestEconomicIntuition(unittest.TestCase):

    def test_higher_price_reduces_grid_buy(self):
        """Doubling electricity price should not increase total grid purchases."""
        _, res_low  = solve(price=FLAT_PRICE,       name="mono_price_low")
        _, res_high = solve(price=FLAT_PRICE * 2.0, name="mono_price_high")
        total_low  = res_low['P_buy'].sum()
        total_high = res_high['P_buy'].sum()
        # Higher price → optimizer shifts away from grid
        self.assertLessEqual(total_high, total_low + ATOL)

    def test_more_solar_reduces_grid_buy(self):
        """More solar availability should not increase grid purchases."""
        _, res_low  = solve(solar=SOLAR_PROFILE * 0.5, name="mono_solar_low")
        _, res_high = solve(solar=SOLAR_PROFILE * 2.0, name="mono_solar_high")
        self.assertLessEqual(
            res_high['P_buy'].sum(),
            res_low['P_buy'].sum() + ATOL
        )

    def test_higher_cop_reduces_hp_electricity(self):
        """Higher COP means the HP needs less electricity for the same heat."""
        cop_low  = np.full(T, 2.0)
        cop_high = np.full(T, 5.0)
        _, res_low  = solve(cop=cop_low,  name="mono_cop_low")
        _, res_high = solve(cop=cop_high, name="mono_cop_high")
        self.assertLessEqual(
            res_high['P_hp_elec'].sum(),
            res_low['P_hp_elec'].sum() + ATOL
        )

    def test_peak_price_grows_pv_or_bat(self):
        """Pronounced peak prices should incentivise PV and / or battery."""
        opt_flat, _ = solve(price=FLAT_PRICE,  name="mono_flat")
        opt_peak, _ = solve(price=PEAK_PRICE,  name="mono_peak")
        combined_flat = opt_flat['C_PV'] + opt_flat['C_bat']
        combined_peak = opt_peak['C_PV'] + opt_peak['C_bat']
        self.assertGreaterEqual(combined_peak, combined_flat - ATOL)

    def test_objective_non_negative(self):
        """Total annual cost must be non-negative."""
        opt, _ = solve(name="mono_obj_sign")
        self.assertGreaterEqual(opt['obj'], -ATOL)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  EDGE-CASE TESTS
# ═════════════════════════════════════════════════════════════════════════════
class TestEdgeCases(unittest.TestCase):

    def test_very_high_price(self):
        """Extreme prices (€10/kWh) must not crash the solver."""
        opt, _ = solve(price=np.full(T, 10.0), name="edge_very_high_price")
        self.assertIsNotNone(opt)

    def test_constant_solar(self):
        """Constant (flat) solar profile – unusual but valid."""
        opt, _ = solve(solar=np.full(T, 3.0), name="edge_flat_solar")
        self.assertIsNotNone(opt)

    def test_single_peak_thermal(self):
        """Single large thermal spike at noon."""
        th = np.ones(T) * 1.0
        th[12] = 50.0
        # This should remain feasible (HP + TES can handle it)
        opt, res = solve(name="edge_single_thermal_spike")
        self.assertIsNotNone(opt)

    def test_zero_cop_raises_or_handles(self):
        """COP of exactly 1 (minimum physical) should still solve."""
        cop_min = np.full(T, 1.001)
        try:
            opt, _ = solve(cop=cop_min, name="edge_min_cop")
            self.assertIsNotNone(opt)
        except Exception as e:
            self.fail(f"COP=1 raised unexpected exception: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  CONSTRAINT-INTEGRITY TESTS
# ═════════════════════════════════════════════════════════════════════════════
class TestConstraintIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.opt, cls.res = solve(price=PEAK_PRICE, name="integrity_check")

    # ---- Electrical balance ----
    def test_electrical_balance(self):
        """At every hour: PV + bat_dis + grid_buy == load + bat_ch + grid_sell + hp_elec."""
        for t in range(T):
            lhs = (self.res['PV_prod'][t]
                   + self.res['P_bat_dis'][t]
                   + self.res['P_buy'][t])
            rhs = (P_LOAD[t]
                   + self.res['P_bat_ch'][t]
                   + self.res['P_sell'][t]
                   + self.res['P_hp_elec'][t])
            self.assertAlmostEqual(lhs, rhs, delta=ATOL,
                                   msg=f"Electrical balance violated at t={t}")

    # ---- Battery SoC dynamics ----
    def test_battery_soc_dynamics(self):
        """SoC[t] == SoC[t-1] + ETA_CH*ch - dis/ETA_DIS."""
        soc = self.res['SoC_bat']
        ch  = self.res['P_bat_ch']
        dis = self.res['P_bat_dis']
        for t in range(T):
            prev = T - 1 if t == 0 else t - 1
            expected = soc[prev] + (ETA_BAT_CH * ch[t] - dis[t] / ETA_BAT_DIS) * dt
            self.assertAlmostEqual(soc[t], expected, delta=ATOL,
                                   msg=f"Battery dynamics violated at t={t}")

    def test_battery_cyclic(self):
        """SoC at end of day == SoC at start of day."""
        self.assertAlmostEqual(
            self.res['SoC_bat'][-1], self.res['SoC_bat'][0], delta=ATOL)

    def test_battery_soc_within_capacity(self):
        """0 ≤ SoC[t] ≤ C_bat at all times."""
        soc = self.res['SoC_bat']
        cap = self.opt['C_bat']
        self.assertTrue(np.all(soc >= -ATOL), f"SoC below 0: {soc.min():.4f}")
        self.assertTrue(np.all(soc <= cap + ATOL), f"SoC above cap: {soc.max():.4f} > {cap:.4f}")

    # ---- TES dynamics ----
    def test_tes_cyclic(self):
        """Q_tes at end of day == Q_tes at start of day."""
        self.assertAlmostEqual(
            self.res['Q_tes'][-1], self.res['Q_tes'][0], delta=ATOL)

    def test_tes_within_bounds(self):
        """0.05·C_TES ≤ Q_tes[t] ≤ 0.95·C_TES."""
        q   = self.res['Q_tes']
        cap = self.opt['C_TES']
        self.assertTrue(np.all(q >= 0.05 * cap - ATOL),
                        f"TES below 5% min: {q.min():.4f}")
        self.assertTrue(np.all(q <= 0.95 * cap + ATOL),
                        f"TES above 95% max: {q.max():.4f} > {0.95*cap:.4f}")

    def test_tes_dynamics(self):
        """Q_tes[t] == Q_tes[t-1] - losses + η_in*Q_hp - P_thermal/η_out."""
        q    = self.res['Q_tes']
        q_hp = self.res['Q_hp_th']
        cap  = self.opt['C_TES']
        for t in range(T):
            prev   = T - 1 if t == 0 else t - 1
            losses = beta * q[prev] + gamma * cap + loss_lids_kwh
            expected = (q[prev] - losses
                        + eta_in * q_hp[t]
                        - P_THERMAL_LOAD[t] / eta_out)
            self.assertAlmostEqual(q[t], expected, delta=ATOL,
                                   msg=f"TES dynamics violated at t={t}")

    # ---- HP capacity ----
    def test_hp_within_capacity(self):
        """HP thermal output ≤ C_HP at all times."""
        q_hp = self.res['Q_hp_th']
        cap  = self.opt['C_HP']
        self.assertTrue(np.all(q_hp <= cap + ATOL),
                        f"HP output exceeds capacity: {q_hp.max():.4f} > {cap:.4f}")

    # ---- Non-negativity ----
    def test_all_flows_non_negative(self):
        for key in ['P_buy', 'P_sell', 'P_bat_ch', 'P_bat_dis', 'P_hp_elec', 'Q_tes']:
            arr = self.res[key]
            self.assertTrue(np.all(arr >= -ATOL),
                            f"{key} has negative values: min={arr.min():.4f}")

    def test_design_variables_non_negative(self):
        for key in ['C_PV', 'C_bat', 'C_HP', 'h_tank', 'C_TES']:
            self.assertGreaterEqual(self.opt[key], -ATOL,
                                    f"{key} is negative: {self.opt[key]:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# 6.  PLOTTING SUITE  (runs all scenarios and saves a multi-panel summary)
# ═════════════════════════════════════════════════════════════════════════════

def run_and_plot_all_scenarios():
    """
    Runs 5 illustrative scenarios and produces a single summary figure
    saved to 'test_scenario_summary.png'.
    """

    scenarios = {
        "S1: Base (peak price + solar)":
            dict(solar=SOLAR_PROFILE, cop=FLAT_COP, price=PEAK_PRICE),
        "S2: No solar, flat price":
            dict(solar=np.zeros(T),   cop=FLAT_COP, price=FLAT_PRICE),
        "S3: No solar, peak price":
            dict(solar=np.zeros(T),   cop=FLAT_COP, price=PEAK_PRICE),
        "S4: Good solar, flat price":
            dict(solar=SOLAR_PROFILE * 2, cop=FLAT_COP, price=FLAT_PRICE),
        "S5: High COP (warm day)":
            dict(solar=SOLAR_PROFILE, cop=np.full(T, calculate_lorenz_cop(temp_c, temp_h, 20.0)),
                 price=PEAK_PRICE),
    }

    results = {}
    for name, kwargs in scenarios.items():
        print(f"\n{'─'*55}")
        print(f"  Plotting scenario: {name}")
        r = run_integrated_optimization(
            kwargs['solar'], kwargs['cop'], kwargs['price'], name)
        if r:
            results[name] = r

    n = len(results)
    fig = plt.figure(figsize=(22, n * 7))
    gs  = gridspec.GridSpec(n, 1, figure=fig, hspace=0.55)

    for row, (name, (opt, res)) in enumerate(results.items()):
        price = scenarios[name]['price']
        cop_t = scenarios[name]['cop']

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs[row], hspace=0.45, wspace=0.35)

        ax_elec  = fig.add_subplot(inner_gs[0, 0])
        ax_therm = fig.add_subplot(inner_gs[0, 1])
        ax_bat   = fig.add_subplot(inner_gs[1, 0])
        ax_price = fig.add_subplot(inner_gs[1, 1])

        subtitle = (f"{name}  |  "
                    f"PV={opt['C_PV']:.1f} kWp, "
                    f"Bat={opt['C_bat']:.1f} kWh, "
                    f"HP={opt['C_HP']:.1f} kW_th, "
                    f"TES h={opt['h_tank']:.2f} m ({opt['C_TES']:.1f} kWh)  |  "
                    f"Cost: €{opt['obj']:,.0f}/yr")
        fig.text(0.5,
                 gs[row].get_position(fig).y1 + 0.005,
                 subtitle,
                 ha='center', fontsize=10, fontweight='bold',
                 transform=fig.transFigure)

        # --- Electrical ---
        ax_elec.stackplot(time,
                          res['PV_prod'],
                          res['P_bat_dis'] - res['P_bat_ch'],
                          res['P_buy'],
                          labels=['PV', 'Battery net', 'Grid buy'],
                          colors=['#f9c74f', '#90be6d', '#4cc9f0'],
                          alpha=0.75)
        ax_elec.plot(time, P_LOAD + res['P_hp_elec'],
                     'k--', lw=2, label='Load + HP')
        ax_elec.set_title('Electrical flows', fontsize=10)
        ax_elec.set_ylabel('kW_e')
        ax_elec.legend(fontsize=8, ncol=2)
        ax_elec.grid(True, alpha=0.3)

        # --- Thermal ---
        ax_therm.fill_between(time, res['Q_hp_th'], alpha=0.5,
                               color='#e63946', label='HP thermal out')
        ax_therm.plot(time, P_THERMAL_LOAD, 'k--', lw=2, label='Thermal load')
        ax2 = ax_therm.twinx()
        ax2.plot(time, res['Q_tes'], color='#1d3557', lw=2.5, label='TES SoC')
        ax2.axhline(0.95 * opt['C_TES'], color='#1d3557', ls=':', alpha=0.5)
        ax2.set_ylabel('kWh (TES)', color='#1d3557', fontsize=9)
        ax_therm.set_title('Thermal system', fontsize=10)
        ax_therm.set_ylabel('kW_th')
        lines1, l1 = ax_therm.get_legend_handles_labels()
        lines2, l2 = ax2.get_legend_handles_labels()
        ax_therm.legend(lines1 + lines2, l1 + l2, fontsize=8)
        ax_therm.grid(True, alpha=0.3)

        # --- Battery SoC ---
        ax_bat.fill_between(time, res['SoC_bat'],
                             alpha=0.5, color='#2a9d8f', label='Battery SoC')
        ax_bat.axhline(opt['C_bat'], color='#2a9d8f', ls='--',
                       alpha=0.7, label=f"Capacity {opt['C_bat']:.1f} kWh")
        ax3 = ax_bat.twinx()
        ax3.plot(time, cop_t, color='#6a0572', lw=2, ls='-.', label='COP')
        ax3.set_ylabel('COP', color='#6a0572', fontsize=9)
        ax_bat.set_title('Battery & COP', fontsize=10)
        ax_bat.set_ylabel('kWh')
        lines1, l1 = ax_bat.get_legend_handles_labels()
        lines2, l2 = ax3.get_legend_handles_labels()
        ax_bat.legend(lines1 + lines2, l1 + l2, fontsize=8)
        ax_bat.grid(True, alpha=0.3)

        # --- Price & TES losses ---
        ax_price.step(time, price, color='#6200b3', lw=2,
                      where='post', label='Buy price')
        ax5 = ax_price.twinx()
        ax5.bar(time, res['TES_loss'], color='#c1121f', alpha=0.45,
                label='TES losses')
        ax5.set_ylabel('kWh losses', color='#c1121f', fontsize=9)
        ax_price.set_title('Electricity price & TES losses', fontsize=10)
        ax_price.set_ylabel('€/kWh')
        ax_price.set_xlabel('Hour')
        lines1, l1 = ax_price.get_legend_handles_labels()
        lines2, l2 = ax5.get_legend_handles_labels()
        ax_price.legend(lines1 + lines2, l1 + l2, fontsize=8)
        ax_price.grid(True, alpha=0.3)

    fig.suptitle('Integrated MILP – Scenario Comparison', fontsize=16,
                 fontweight='bold', y=1.002)
    out_path = 'test_scenario_summary.png'
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\n✅  Summary plot saved → {out_path}")
    plt.close(fig)

    # ── Capacity bar chart ──────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 5))
    labels   = [f"S{i+1}" for i in range(len(results))]
    c_pv     = [r[0]['C_PV']   for r in results.values()]
    c_bat    = [r[0]['C_bat']  for r in results.values()]
    c_hp     = [r[0]['C_HP']   for r in results.values()]
    h_tank   = [r[0]['h_tank'] for r in results.values()]

    for ax, vals, ylabel, colour, title in zip(
            axes2,
            [c_pv, c_bat, c_hp, h_tank],
            ['kWp', 'kWh', 'kW_th', 'm'],
            ['#f9c74f', '#2a9d8f', '#e63946', '#1d3557'],
            ['PV Capacity', 'Battery Capacity', 'HP Capacity', 'TES Height']):
        bars = ax.bar(labels, vals, color=colour, edgecolor='black', alpha=0.85)
        ax.bar_label(bars, fmt='%.2f', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)

    fig2.suptitle('Optimal Capacities per Scenario', fontsize=14, fontweight='bold')
    plt.tight_layout()
    cap_path = 'test_capacity_comparison.png'
    fig2.savefig(cap_path, dpi=130, bbox_inches='tight')
    print(f"✅  Capacity comparison saved → {cap_path}")
    plt.close(fig2)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("  INTEGRATED MILP – TEST SUITE")
    print("=" * 60)

    # 1. Unit + solver tests
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    for cls in [TestPhysicsHelpers, TestFeasibility,
                TestEconomicIntuition, TestEdgeCases,
                TestConstraintIntegrity]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    # 2. Scenario plots (always run, even if some tests fail)
    print("\n" + "=" * 60)
    print("  GENERATING SCENARIO PLOTS")
    print("=" * 60)
    run_and_plot_all_scenarios()

    # Exit with error code if any test failed
    sys.exit(0 if result.wasSuccessful() else 1)