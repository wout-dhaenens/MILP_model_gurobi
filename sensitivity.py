"""
OAT (One-At-a-Time) Sensitivity Analysis — Spider Plot
=======================================================
Wraps run_integrated_optimization() and varies one parameter at a time
across [-30%, -20%, -10%, 0%, +10%, +20%, +30%] from its baseline value.

Outputs tracked per solve:
  - Total annualised cost   (objective value)
  - Optimal PV capacity     [kWp]
  - Optimal battery size    [kWh]
  - Optimal HP capacity     [kW_th]
  - Optimal TES size        [kWh]

Results are saved to:
  - sensitivity_results.csv      (raw numbers)
  - spider_plot_<output>.png     (one spider plot per output metric)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy, time, os
import math

# ── Import your existing model ────────────────────────────────────────────────
# Adjust the import path if needed
from Milp_yearly_test import (
    run_integrated_optimization,
    T, dt,
    CAPEX_BAT_ANNUAL, CAPEX_PV_ANNUAL, CAPEX_HP_ANNUAL, CAPEX_TES_eu_L,
    C_CAP, ETA_BAT_CH, ETA_BAT_DIS,
    Factor_Thermal, P_THERMAL_LOAD, P_LOAD,
    load_pvgis_from_csv, load_prices_from_csv,
)

# ==============================================================================
# 1.  LOAD FIXED INPUTS (done once)
# ==============================================================================
print("Loading solar, COP and price data …")
I_SOLAR_BASE, COP_t_BASE = load_pvgis_from_csv()
P_price_buy_BASE, P_price_sell_BASE = load_prices_from_csv()

# ==============================================================================
# 2.  DEFINE PARAMETERS TO VARY
# ==============================================================================
# Each entry: (display_name, internal_key, baseline_value)
# internal_key is used inside the solve wrapper to apply the perturbation.

PARAMETERS = [
    #("CAPEX PV [€/kWp/y]",          "CAPEX_PV",       CAPEX_PV_ANNUAL),   # ← active for test run
     ("CAPEX Battery [€/kWh/y]",      "CAPEX_BAT",      CAPEX_BAT_ANNUAL),
    # ("CAPEX HP [€/kW_th/y]",         "CAPEX_HP",       CAPEX_HP_ANNUAL),
     ("CAPEX TES [€/L/y]",            "CAPEX_TES",      CAPEX_TES_eu_L),
     ("Elec. buy price [scale]",       "PRICE_SCALE",    1.0),
    # ("Capacity tariff C_CAP [€/kW/mo]","C_CAP",        C_CAP),
     ("Thermal demand factor",         "FACTOR_THERMAL", Factor_Thermal),
    # ("Battery efficiency η_rt",       "ETA_BAT",        ETA_BAT_CH * ETA_BAT_DIS),
]

PERTURBATIONS = np.array([-0.30, 0 , 0.30])#np.array([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30])  # relative change

OUTPUT_KEYS   = ["obj", "C_PV", "C_bat", "C_HP", "C_TES"]
OUTPUT_LABELS = {
    "obj":   "Total Annualised Cost [€/yr]",
    "C_PV":  "Optimal PV Capacity [kWp]",
    "C_bat": "Optimal Battery Size [kWh]",
    "C_HP":  "Optimal HP Capacity [kW_th]",
    "C_TES": "Optimal TES Size [kWh]",
}

# ==============================================================================
# 3.  SOLVE WRAPPER — applies a single parameter perturbation
# ==============================================================================

def solve_with_perturbation(param_key: str, factor: float):
    """
    Run the MILP with one parameter scaled by (1 + factor).
    Returns a dict with keys matching OUTPUT_KEYS, or None on failure.

    'factor' is the *relative* change: 0.0 = baseline, 0.10 = +10 %, etc.
    """
    import Milp_yearly_test as m   # re-import so we can monkey-patch module globals

    scale = 1.0 + factor

    # ── Patch module-level globals that the optimisation reads ────────────────
    original = {}

    def patch(attr, value):
        original[attr] = getattr(m, attr)
        setattr(m, attr, value)

    if param_key == "CAPEX_PV":
        patch("CAPEX_PV_ANNUAL", m.CAPEX_PV_ANNUAL * scale)

    elif param_key == "CAPEX_BAT":
        patch("CAPEX_BAT_ANNUAL", m.CAPEX_BAT_ANNUAL * scale)

    elif param_key == "CAPEX_HP":
        patch("CAPEX_HP_ANNUAL", m.CAPEX_HP_ANNUAL * scale)

    elif param_key == "CAPEX_TES":
        # CAPEX_TES_eu_L feeds into CAPEX_TES_METER_ANNUAL and capex_TES_annual
        new_eu_L = m.CAPEX_TES_eu_L * scale
        patch("CAPEX_TES_eu_L", new_eu_L)
        new_meter = new_eu_L * np.pi * m.d**2 / 4 * 1000
        patch("CAPEX_TES_METER_ANNUAL", new_meter)

    elif param_key == "C_CAP":
        patch("C_CAP", m.C_CAP * scale)

    elif param_key == "FACTOR_THERMAL":
        # Rescale both demand profiles (thermal and electrical = 10 % of thermal)
        new_factor = m.Factor_Thermal * scale
        patch("Factor_Thermal", new_factor)
        from Milp_yearly_test import load_demand_profiles, THERMAL_XLSX
        new_thermal, new_elec = load_demand_profiles(THERMAL_XLSX)
        # load_demand_profiles already applies Factor_Thermal internally,
        # but Factor_Thermal has been patched, so we reload:
        patch("P_THERMAL_LOAD", new_thermal)
        patch("P_LOAD",         new_elec)

    elif param_key == "ETA_BAT":
        # Keep charge and discharge efficiencies equal: η_ch = η_dis = √(η_rt·scale)
        eta = np.sqrt(m.ETA_BAT_CH * m.ETA_BAT_DIS * scale)
        eta = min(eta, 0.999)
        patch("ETA_BAT_CH",  eta)
        patch("ETA_BAT_DIS", eta)

    # ── Choose price arrays ────────────────────────────────────────────────────
    if param_key == "PRICE_SCALE":
        buy  = P_price_buy_BASE  * scale
        sell = P_price_sell_BASE * scale
    else:
        buy  = P_price_buy_BASE.copy()
        sell = P_price_sell_BASE.copy()

    # ── Solve ─────────────────────────────────────────────────────────────────
    try:
        opt, res = m.run_integrated_optimization(I_SOLAR_BASE, COP_t_BASE, buy, sell)
    except Exception as e:
        print(f"    [ERROR] {param_key} factor={factor:+.2f}: {e}")
        opt = None

    # ── Restore patched globals ───────────────────────────────────────────────
    for attr, val in original.items():
        setattr(m, attr, val)

    if opt is None:
        return None

    return {
        "obj":   opt["obj"],
        "C_PV":  opt["C_PV"],
        "C_bat": opt["C_bat"],
        "C_HP":  opt["C_HP"],
        "C_TES": opt["C_TES"],
    }


# ==============================================================================
# 4.  RUN THE OAT SWEEP
# ==============================================================================

# results[param_key][output_key] = array of len(PERTURBATIONS)
results = {pk: {ok: np.full(len(PERTURBATIONS), np.nan) for ok in OUTPUT_KEYS}
           for _, pk, _ in PARAMETERS}

total_solves = len(PARAMETERS) * len(PERTURBATIONS)
solve_count  = 0
t_start      = time.time()

print(f"\n{'='*60}")
print(f"Starting OAT sweep: {total_solves} MILP solves")
print(f"{'='*60}\n")

for param_label, param_key, param_baseline in PARAMETERS:
    print(f"\n── Parameter: {param_label} (baseline = {param_baseline:.4g}) ──")
    for i, pct in enumerate(PERTURBATIONS):
        solve_count += 1
        elapsed = time.time() - t_start
        print(f"  [{solve_count}/{total_solves}] {pct:+.0%}  (elapsed: {elapsed/60:.1f} min)")

        out = solve_with_perturbation(param_key, pct)
        if out is not None:
            for ok in OUTPUT_KEYS:
                results[param_key][ok][i] = out[ok]
        else:
            print(f"    !! Solve failed — NaN recorded")

print(f"\nAll solves done in {(time.time()-t_start)/60:.1f} min\n")


# ==============================================================================
# 5.  SAVE RAW RESULTS TO CSV
# ==============================================================================

rows = []
for param_label, param_key, param_baseline in PARAMETERS:
    for i, pct in enumerate(PERTURBATIONS):
        row = {"parameter": param_label, "perturbation_pct": pct * 100}
        for ok in OUTPUT_KEYS:
            row[ok] = results[param_key][ok][i]
        rows.append(row)

df_results = pd.DataFrame(rows)
df_results.to_csv("sensitivity_results.csv", index=False)
print("Raw results saved → sensitivity_results.csv")


# ==============================================================================
# 6.  SPIDER PLOTS — one per output metric
# ==============================================================================

def make_spider_plots_combined(output_items):
    """
    Plots all spider plots in one combined figure with subplots.
    """
    n = len(output_items)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = np.array(axes).flatten()  # ensure iterable even if 1 row

    colors = cm.tab10(np.linspace(0, 0.9, len(PARAMETERS)))
    x_pct = PERTURBATIONS * 100

    for idx, (output_key, output_label) in enumerate(output_items):
        ax = axes[idx]
        for (param_label, param_key, _), color in zip(PARAMETERS, colors):
            y_abs    = results[param_key][output_key]
            baseline = y_abs[PERTURBATIONS == 0.0][0]
            if np.isnan(baseline) or baseline == 0:
                print(f"  Skipping {param_label} for {output_key} (baseline is NaN or 0)")
                continue
            y_pct = (y_abs - baseline) / abs(baseline) * 100
            ax.plot(x_pct, y_pct,
                    marker='o', linewidth=2, markersize=5,
                    label=param_label, color=color)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel("Parameter change from baseline [%]", fontsize=10)
        ax.set_ylabel(f"Change in {output_label} [%]", fontsize=10)
        ax.set_title(f"OAT Sensitivity — {output_label}", fontsize=11, fontweight='bold')
        ax.set_xticks(x_pct)
        ax.grid(True, alpha=0.3)

    # Shared legend from last active axis
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               ncol=min(len(PARAMETERS), 4),
               fontsize=9,
               bbox_to_anchor=(0.5, -0.02),
               borderaxespad=0)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("OAT Sensitivity Analysis — All Outputs", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = "spider_plots_combined.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Combined spider plot saved → {fname}")

make_spider_plots_combined(list(OUTPUT_LABELS.items()))