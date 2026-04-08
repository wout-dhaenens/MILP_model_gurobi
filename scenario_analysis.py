"""
scenario_analysis.py
====================
Parametric sweep over key MILP parameters.  Each scenario group runs a sweep
over one parameter while keeping everything else at its baseline value.

── HOW TO RUN IN PARALLEL (Spyder) ─────────────────────────────────────────────
Open one IPython console per group you want to run simultaneously, then in each:

    runfile('scenario_analysis.py', args='capex_bat')
    runfile('scenario_analysis.py', args='capex_pv')
    runfile('scenario_analysis.py', args='capex_hp')
    runfile('scenario_analysis.py', args='capex_ltes')
    runfile('scenario_analysis.py', args='capex_wtes')
    runfile('scenario_analysis.py', args='discount')
    runfile('scenario_analysis.py', args='max_capex')
    runfile('scenario_analysis.py', args='cap_tariff')
    runfile('scenario_analysis.py', args='elec_price')
    runfile('scenario_analysis.py', args='gas_year')
    runfile('scenario_analysis.py', args='temp_h')
    runfile('scenario_analysis.py', args='cop_scale')

── SCENARIO KINDS ───────────────────────────────────────────────────────────────
  "capex"       CAPEX base parameter → recomputes all derived globals + PWL BPs
  "simple"      Module global with no downstream derived values (C_CAP, etc.)
  "price_scale" Multiplies both price arrays before passing them to the solver
  "gas_year"    Reloads P_THERMAL_LOAD from the gas CSV for a different year
  "temp_h"      Changes supply temperature; temp_c = temp_h - 20 always;
                recomputes WTES loss params and corrects COP via Carnot ratio
  "cop_scale"   Multiplies the entire COP_t array by a factor

── OUTPUT ───────────────────────────────────────────────────────────────────────
Results are written to:   ./scenario_results/scenario_<group>.csv
"""

import sys
import os
import numpy as np
import pandas as pd

# ==============================================================================
# 0.  Select scenario group
# ==============================================================================
SCENARIO_GROUP = sys.argv[1] if len(sys.argv) > 1 else "capex_bat"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenario_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1.  Import the MILP module and shared data loaders
# ==============================================================================
import Milp_yearly_test_gurobi as M
from Fetch_and_save_data import (load_pvgis_from_csv, load_prices_from_csv,
                                  load_prices_from_epex, fetch_pvgis_to_csv)

# ==============================================================================
# 2.  Load all time-series data once  (shared across every run in this console)
# ==============================================================================
print("Loading shared input data …")
fetch_pvgis_to_csv(M.temp_c, M.temp_h)
I_SOLAR, COP_t_base, T_amb = load_pvgis_from_csv()

if M.PRICE_SOURCE == "epex":
    P_price_buy_base, P_price_sell_base = load_prices_from_epex(M.EPEX_CSV_PATH)
else:
    P_price_buy_base, P_price_sell_base = load_prices_from_csv()

Cap_frac_t = M.calculate_capacity_fraction(T_amb)

assert len(P_price_buy_base) == M.T
assert len(I_SOLAR)          == M.T
assert len(COP_t_base)       == M.T
assert len(Cap_frac_t)       == M.T
print("Data loaded OK.\n")

# Baseline supply temperature (read once so teardown can always restore it)
TEMP_H_BASELINE = float(M.temp_h)   # 60 °C

# ==============================================================================
# 3.  Scenario definitions
#
#   kind   → how the scenario is applied (see module docstring)
#   param  → human-readable label stored in CSV
#   values → list of values to sweep
#   unit   → unit label stored in CSV
# ==============================================================================
SCENARIOS = {
    # ── CAPEX sweeps ──────────────────────────────────────────────────────────
    "capex_bat":  dict(kind="capex", param="CAPEX_BAT_UNIT",      values=[400, 600, 800, 1000, 1200],          unit="eu_per_kWh"),
    "capex_pv":   dict(kind="capex", param="CAPEX_PV_UNIT",       values=[500, 700, 900, 1100, 1300],          unit="eu_per_kWp"),
    "capex_hp":   dict(kind="capex", param="CAPEX_HP_UNIT",       values=[4000, 5500, 7000, 8500, 10000],      unit="eu_per_kW_th"),
    "capex_ltes": dict(kind="capex", param="CAPEX_LTES_UNIT",     values=[300, 450, 600, 750, 900],            unit="eu_per_kWh_th"),
    "capex_wtes": dict(kind="capex", param="WTES_REF_COST_EUR_L", values=[1.5, 2.0, 3.0, 4.0, 5.0],           unit="eu_per_L"),
    "discount":   dict(kind="capex", param="DISCOUNT_RATE",       values=[0.02, 0.03, 0.05, 0.07, 0.10],      unit="fraction"),
    "max_capex":  dict(kind="capex", param="MAX_ANNUAL_CAPEX",    values=[100e3, 150e3, 200e3, 250e3, 300e3], unit="eu_per_yr"),

    # ── OPEX / tariff sweeps ──────────────────────────────────────────────────
    "cap_tariff": dict(kind="simple", param="C_CAP",              values=[2, 3, 5, 7, 10],                    unit="eu_per_kW_month"),

    # ── Electricity price level ───────────────────────────────────────────────
    "elec_price": dict(kind="price_scale", param="price_multiplier", values=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0], unit="fraction"),

    # ── Thermal demand year ───────────────────────────────────────────────────
    "gas_year":   dict(kind="gas_year", param="GAS_DEMAND_YEAR",  values=[2019, 2020, 2021, 2022, 2023],       unit="yr"),

    # ── Technical / operational ───────────────────────────────────────────────
    # Supply temperature; temp_c is always kept at temp_h - 20 °C.
    # COP is corrected via Carnot ratio relative to the baseline (60 °C).
    # WTES loss parameters (gamma_wtes, loss_lids_wtes) are recomputed accordingly.
    "temp_h":     dict(kind="temp_h",    param="temp_h",          values=[45, 50, 55, 60, 65],                unit="degC"),

    # Uniform multiplier on the full COP_t profile — captures a better/worse HP
    # model or a different climate zone without re-fetching PVGIS data.
    "cop_scale":  dict(kind="cop_scale", param="COP_multiplier",  values=[0.8, 0.9, 1.0, 1.1, 1.2],          unit="fraction"),
}

if SCENARIO_GROUP not in SCENARIOS:
    raise ValueError(
        f"Unknown scenario group '{SCENARIO_GROUP}'.\n"
        f"Available groups:\n" +
        "\n".join(f"  {k:12s}  ({v['kind']})  {v['param']}  [{v['unit']}]"
                  for k, v in SCENARIOS.items())
    )

# ==============================================================================
# 4.  CAPEX patch helper
# ==============================================================================

def patch_and_recompute(module, param_name, value):
    """
    Set module.<param_name> = value, then recompute every downstream derived
    global (annual CAPEX rates and PWL breakpoints).
    """
    setattr(module, param_name, value)

    crf = module.crf
    dr  = module.DISCOUNT_RATE

    module.CAPEX_BAT_ANNUAL      = module.CAPEX_BAT_UNIT  * crf(dr, module.LIFETIME_BAT)
    module.CAPEX_PV_ANNUAL       = module.CAPEX_PV_UNIT   * crf(dr, module.LIFETIME_PV)
    module.CAPEX_HP_ANNUAL       = module.CAPEX_HP_UNIT   * crf(dr, module.LIFETIME_HP)
    module.CAPEX_TES_eu_kWh_LTES = module.CAPEX_LTES_UNIT * crf(dr, module.LIFETIME_TES)
    module.CAPEX_TES_eu_kWh_WTES = (
        module.WTES_REF_COST_EUR_L * 1000.0
        / ((module.rho_water * module.c_water * module.delta_T_HC) / 3.6e6)
    ) * crf(dr, module.LIFETIME_TES)
    module.CAPEX_WTES_METER_ANNUAL = module.CAPEX_TES_eu_kWh_WTES * module.kWh_per_m_wtes

    gen = module.generate_capex_pwl
    module.BP_X_PV,   module.BP_Y_PV   = gen(module.C_PV_MAX,   module.CAPEX_PV_ANNUAL,        anchor_cap=10.0)
    module.BP_X_BAT,  module.BP_Y_BAT  = gen(module.C_BAT_MAX,  module.CAPEX_BAT_ANNUAL,       anchor_cap=10.0)
    module.BP_X_HP,   module.BP_Y_HP   = gen(module.C_HP_MAX,   module.CAPEX_HP_ANNUAL,        anchor_cap=30.0)
    module.BP_X_LTES, module.BP_Y_LTES = gen(module.C_LTES_MAX, module.CAPEX_TES_eu_kWh_LTES, anchor_cap=30.0)
    module.BP_X_WTES, module.BP_Y_WTES = gen(module.C_WTES_MAX, module.CAPEX_TES_eu_kWh_WTES, anchor_cap=module.WTES_REF_KWH)

# ==============================================================================
# 5.  Temperature patch helper
# ==============================================================================

def patch_temp_h(module, t_h_new):
    """
    Set temp_h and temp_c = temp_h - 20, then recompute all derived WTES
    loss parameters.  delta_T_HC stays at 20 K so kWh_per_m_wtes and all
    CAPEX breakpoints are unaffected.
    """
    t_c_new = t_h_new - 20.0

    module.temp_h      = t_h_new
    module.temp_c      = t_c_new
    module.T_HIGH      = t_h_new
    module.T_LOW       = t_c_new
    # delta_T_HC = 20 always → no change needed
    module.delta_T_C0  = t_c_new - module.temp_env
    module.delta_T_H0  = t_h_new - module.temp_env

    module.gamma_wtes = module.beta_wtes * (module.delta_T_C0 / module.delta_T_HC)
    module.loss_lids_wtes = (
        module.U_wall_wtes * 2.0 * module.A_cross_wtes
        * ((module.delta_T_H0 + module.delta_T_C0) / 2.0)
        * module.dt_sec / 3.6e6
    )


def cop_carnot_correction(COP_base, T_amb_arr, T_h_base, T_h_new):
    """
    Scale a COP array for a new supply temperature using the Carnot ratio.

        COP_new ≈ COP_base × [COP_Carnot(T_h_new)] / [COP_Carnot(T_h_base)]

    where  COP_Carnot = T_h_K / (T_h_K - T_amb_K).

    All temperatures in °C; converted to K internally.
    """
    T_h_base_K = T_h_base + 273.15
    T_h_new_K  = T_h_new  + 273.15
    T_amb_K    = T_amb_arr + 273.15

    carnot_base = T_h_base_K / (T_h_base_K - T_amb_K)
    carnot_new  = T_h_new_K  / (T_h_new_K  - T_amb_K)

    return np.clip(COP_base * (carnot_new / carnot_base), a_min=1.0, a_max=None)

# ==============================================================================
# 6.  setup_run — apply the scenario mutation and return all 5 solver inputs
# ==============================================================================

SIMPLE_DEFAULTS = {
    "C_CAP": 5.0,
}

def setup_run(scenario, value):
    """
    Mutate module state for this scenario value.
    Returns (I_SOLAR_use, COP_t_use, Cap_frac_t_use, p_buy, p_sell)
    ready to pass directly to run_integrated_optimization().
    """
    kind = scenario["kind"]
    param = scenario["param"]

    # defaults — most scenarios leave these unchanged
    COP_t_use = COP_t_base.copy()
    p_buy     = P_price_buy_base
    p_sell    = P_price_sell_base

    if kind == "capex":
        patch_and_recompute(M, param, value)

    elif kind == "simple":
        setattr(M, param, value)

    elif kind == "price_scale":
        p_buy  = P_price_buy_base  * value
        p_sell = P_price_sell_base * value

    elif kind == "gas_year":
        print(f"  Reloading gas demand for year {value} …")
        M.P_THERMAL_LOAD = M.load_demand_from_gascsv(M.GAS_CSV_PATH, value)
        print(f"  Peak = {max(M.P_THERMAL_LOAD):.2f} kW_th  |  "
              f"Annual = {sum(M.P_THERMAL_LOAD):.0f} kWh_th")

    elif kind == "temp_h":
        patch_temp_h(M, value)
        COP_t_use = cop_carnot_correction(COP_t_base, T_amb, TEMP_H_BASELINE, value)
        print(f"  temp_c = {value - 20:.0f} °C  |  "
              f"COP mean: {COP_t_base.mean():.2f} → {COP_t_use.mean():.2f}")

    elif kind == "cop_scale":
        COP_t_use = COP_t_base * value
        print(f"  COP mean: {COP_t_base.mean():.2f} → {COP_t_use.mean():.2f}")

    else:
        raise ValueError(f"Unknown scenario kind: '{kind}'")

    return I_SOLAR, COP_t_use, Cap_frac_t, p_buy, p_sell


def teardown_run(scenario, value):
    """Restore any module-level state so the next iteration starts clean."""
    kind  = scenario["kind"]
    param = scenario["param"]

    if kind == "capex":
        pass  # overwritten correctly at the start of the next iteration

    elif kind == "simple":
        if param in SIMPLE_DEFAULTS:
            setattr(M, param, SIMPLE_DEFAULTS[param])

    elif kind == "price_scale":
        pass  # base arrays were never modified

    elif kind == "gas_year":
        M.P_THERMAL_LOAD = M.load_demand_from_gascsv(M.GAS_CSV_PATH, M.GAS_DEMAND_YEAR)

    elif kind == "temp_h":
        patch_temp_h(M, TEMP_H_BASELINE)

    elif kind == "cop_scale":
        pass  # COP_t_base was never modified

# ==============================================================================
# 7.  Helper: interpolate PWL cost for a given capacity
# ==============================================================================

def pwl_eval(x, bp_x, bp_y):
    return float(np.interp(x, bp_x, bp_y))

# ==============================================================================
# 8.  Helper: flatten one solver result into a CSV row dict
# ==============================================================================

def extract_row(scenario, value, opt, res, COP_t_used, p_buy_used):
    param = scenario["param"]
    unit  = scenario["unit"]

    capex_pv   = pwl_eval(opt["C_PV"],   M.BP_X_PV,   M.BP_Y_PV)
    capex_bat  = pwl_eval(opt["C_bat"],  M.BP_X_BAT,  M.BP_Y_BAT)
    capex_hp   = pwl_eval(opt["C_HP"],   M.BP_X_HP,   M.BP_Y_HP)
    capex_ltes = pwl_eval(opt["C_LTES"], M.BP_X_LTES, M.BP_Y_LTES)
    capex_wtes = pwl_eval(opt["C_WTES"], M.BP_X_WTES, M.BP_Y_WTES)
    capex_tes  = capex_ltes + capex_wtes
    capex_tot  = capex_pv + capex_bat + capex_hp + capex_tes
    opex_net   = opt["obj"] - capex_tot

    if opt["y_ltes"] and opt["C_LTES"] > 0:
        capex_tes_eu_kwh = capex_tes / opt["C_LTES"]
    elif opt["C_WTES"] > 0:
        capex_tes_eu_kwh = capex_tes / opt["C_WTES"]
    else:
        capex_tes_eu_kwh = 0.0

    return {
        # Scenario identification
        "scenario_group":         SCENARIO_GROUP,
        "param":                  param,
        "param_unit":             unit,
        "param_value":            value,

        # Sizing
        "C_PV_kWp":               opt["C_PV"],
        "C_bat_kWh":              opt["C_bat"],
        "C_HP_kW_th":             opt["C_HP"],
        "TES_type":               opt["tes_type"],
        "C_TES_kWh":              opt["C_TES"],
        "V_TES_m3":               opt["V_tes"],

        # CAPEX (€/yr, annualised + EOS-adjusted)
        "capex_pv_eu_yr":         capex_pv,
        "capex_bat_eu_yr":        capex_bat,
        "capex_hp_eu_yr":         capex_hp,
        "capex_tes_eu_yr":        capex_tes,
        "capex_total_eu_yr":      capex_tot,
        "capex_tes_eu_kwh_yr":    capex_tes_eu_kwh,

        # Costs (€/yr)
        "opex_net_eu_yr":         opex_net,
        "total_cost_eu_yr":       opt["obj"],

        # Electricity price context
        "price_buy_mean_eu_kWh":  float(np.mean(p_buy_used)),
        "price_buy_max_eu_kWh":   float(np.max(p_buy_used)),

        # COP context
        "COP_mean":               float(np.mean(COP_t_used)),
        "COP_min":                float(np.min(COP_t_used)),

        # Temperature context
        "temp_h_degC":            float(M.temp_h),
        "temp_c_degC":            float(M.temp_c),

        # Energy flows (kWh/yr)
        "E_buy_kWh":              sum(res["P_buy"]),
        "E_sell_kWh":             sum(res["P_sell"]),
        "E_pv_kWh":               sum(res["PV_prod"]),
        "annual_elec_demand_kWh": sum(M.P_LOAD),
        "Q_tes_in_kWh":           sum(res["Q_tes_in"]),
        "Q_tes_out_kWh":          sum(res["Q_tes_out"]),
        "TES_loss_kWh":           sum(res["TES_loss"]),

        # Thermal demand
        "peak_thermal_kW":        max(M.P_THERMAL_LOAD),
        "annual_thermal_kWh":     sum(M.P_THERMAL_LOAD),

        # Grid peak
        "peak_grid_kW":           opt["P_peak_annual"],
    }

# ==============================================================================
# 9.  Run the sweep
# ==============================================================================
scenario = SCENARIOS[SCENARIO_GROUP]
param    = scenario["param"]
values   = scenario["values"]
unit     = scenario["unit"]
kind     = scenario["kind"]

print(f"{'='*62}")
print(f"  Scenario group : {SCENARIO_GROUP}")
print(f"  Kind           : {kind}")
print(f"  Parameter      : {param}  [{unit}]")
print(f"  Values         : {values}")
print(f"{'='*62}\n")

rows = []
for i, v in enumerate(values):
    print(f"\n[{i+1}/{len(values)}]  {param} = {v} {unit}")
    print("-" * 50)

    I_sol, COP_use, Cap_frac, p_buy, p_sell = setup_run(scenario, v)

    opt, res = M.run_integrated_optimization(I_sol, COP_use, Cap_frac, p_buy, p_sell)

    if opt is None:
        print(f"  !! Solver failed — skipping this point.")
        teardown_run(scenario, v)
        continue

    row = extract_row(scenario, v, opt, res, COP_use, p_buy)
    rows.append(row)

    print(f"  → Total cost: €{opt['obj']:>10,.0f}/yr  |  "
          f"PV={opt['C_PV']:.1f} kWp  BAT={opt['C_bat']:.1f} kWh  "
          f"HP={opt['C_HP']:.1f} kW  TES={opt['C_TES']:.1f} kWh ({opt['tes_type']})")

    teardown_run(scenario, v)

# ==============================================================================
# 10.  Save results
# ==============================================================================
if rows:
    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, f"scenario_{SCENARIO_GROUP}.csv")
    df.to_csv(out_path, index=False, float_format="%.4f")

    print(f"\n{'='*62}")
    print(f"  ✓  Saved {len(df)} rows  →  {out_path}")
    print(f"{'='*62}\n")
    print(df.to_string(index=False))
else:
    print("\nNo successful runs — nothing saved.")
