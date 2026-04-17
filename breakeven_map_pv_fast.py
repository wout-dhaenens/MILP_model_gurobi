# -*- coding: utf-8 -*-
"""
breakeven_map_pv_fast.py
========================
Same 2-D breakeven sweep as breakeven_map_pv.py but uses fast=True in
run_integrated_optimization, which removes the 8760 u_hp binary variables
and the C_hp_active auxiliary variables (continuous LP relaxation for the HP).
This makes each solve significantly faster at the cost of omitting the
part-load efficiency penalty and the minimum-load floor for the heat pump.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator

FOLDER = os.path.dirname(os.path.abspath(__file__))
if FOLDER not in sys.path:
    sys.path.insert(0, FOLDER)

import Milp_yearly_test_gurobi as m
import fetch_solcast_data as _scd
from Fetch_and_save_data import load_prices_from_csv
from demand_generation import generate_yearly_bdew_profile, load_bdew_demand_from_csv

plt.style.use('seaborn-v0_8-whitegrid')

# ==============================================================================
# GRID & SETTINGS
# ==============================================================================
CAPEX_RANGE = np.linspace(150, 900, 10)   # EUR/kWp — variable component
PRICE_RANGE = np.linspace(0.4,  2.0, 10)  # price multiplier

BASELINE_CAPEX_VAR = 532.0
BASELINE_MULT      = 1.0

INSTALL_THRESHOLD  =  2.0
SATURATE_THRESHOLD = 200 - 2    # kWp

SWEEP_MIP_GAP    = 0.05
SWEEP_TIME_LIMIT = 3600 / 2     # s per solve

CACHE_FILE = os.path.join(FOLDER, "breakeven_map_results_fast.npy")

# ==============================================================================
# SHARED DATA LOADING
# ==============================================================================
print("=" * 60)
print("  PV Breakeven Map (fast mode) — loading shared inputs")
print("=" * 60)

_scd.YEAR       = m.WEATHER_YEAR
_scd.T          = m.T
_scd.ETA_LORENZ = m.ETA_LORENZ

solcast_csv = os.path.join(FOLDER, f"solcast_data_{m.WEATHER_YEAR}.csv")
I_SOLAR, _, T_amb = _scd.load_solcast_from_csv(solcast_csv)
COP_t      = np.array([_scd.calculate_lorenz_cop(m.temp_c, m.temp_h, t) for t in T_amb])
Cap_frac_t = m.calculate_capacity_fraction(T_amb)

prices_csv = os.path.join(FOLDER, f"prices_data_{m.PRICE_YEAR}.csv")
P_buy_base, P_sell_base = load_prices_from_csv(prices_csv)

generate_yearly_bdew_profile(m.WEATHER_YEAR)
m.P_THERMAL_LOAD = load_bdew_demand_from_csv(m.WEATHER_YEAR)

print(f"\n  Weather year : {m.WEATHER_YEAR}  ({m.T} timesteps)")
print(f"  Price year   : {m.PRICE_YEAR}")
print(f"  Grid size    : {len(CAPEX_RANGE)} x {len(PRICE_RANGE)} = "
      f"{len(CAPEX_RANGE) * len(PRICE_RANGE)} solves")
print(f"  MIP gap : {SWEEP_MIP_GAP*100:.0f}%  |  Time limit : {SWEEP_TIME_LIMIT}s/solve")
print(f"  Mode    : fast=True  (no u_hp binaries, no part-load penalty)\n")

# ==============================================================================
# CACHE CHECK / RUN SWEEP
# ==============================================================================
FROM_CACHE = False

if os.path.exists(CACHE_FILE):
    try:
        saved = np.load(CACHE_FILE, allow_pickle=True).item()
        if (np.allclose(saved.get("capex_range", []), CAPEX_RANGE) and
                np.allclose(saved.get("price_range", []), PRICE_RANGE)):
            C_PV_grid   = saved["C_PV_grid"]
            MIPGap_grid = saved.get("MIPGap_grid",
                                    np.full_like(C_PV_grid, np.nan))
            remaining = int(np.isnan(C_PV_grid).sum())
            if remaining == 0:
                FROM_CACHE = True
                print(f"Loaded cached results from '{os.path.basename(CACHE_FILE)}'")
            else:
                print(f"Partial cache found ({remaining} solves remaining) — resuming sweep.")
        else:
            print("Cache grid definition differs — re-running sweep.")
    except Exception as e:
        print(f"Cache read failed ({e}) — re-running sweep.")

if not FROM_CACHE:
    n_cap = len(CAPEX_RANGE)
    n_pri = len(PRICE_RANGE)
    if "C_PV_grid" not in dir():
        C_PV_grid   = np.full((n_pri, n_cap), np.nan)
        MIPGap_grid = np.full((n_pri, n_cap), np.nan)

    total     = n_cap * n_pri
    remaining = int(np.isnan(C_PV_grid).sum())
    done      = total - remaining
    t_start   = time.time()

    _checkpoint = {"C_PV_grid": C_PV_grid, "MIPGap_grid": MIPGap_grid,
                   "capex_range": CAPEX_RANGE, "price_range": PRICE_RANGE}

    print(f"Starting sweep ({remaining} of {total} solves remaining)...\n")
    for i, price_mult in enumerate(PRICE_RANGE):
        for j, capex_var in enumerate(CAPEX_RANGE):
            if not np.isnan(C_PV_grid[i, j]):
                continue

            done += 1

            m.PV_CAPEX_VAR_ANNUAL = capex_var * m.crf(m.DISCOUNT_RATE, m.LIFETIME_PV)

            P_buy  = P_buy_base  * price_mult
            P_sell = P_sell_base * price_mult

            t0 = time.time()
            try:
                opt, _ = m.run_integrated_optimization(
                    I_SOLAR, COP_t, Cap_frac_t, P_buy, P_sell,
                    mip_gap=SWEEP_MIP_GAP,
                    time_limit=SWEEP_TIME_LIMIT,
                    output_flag=0,
                    fast=True,
                )
                cpv = opt['C_PV']    if opt is not None else np.nan
                gap = opt['mip_gap'] if opt is not None else np.nan
            except Exception as exc:
                print(f"    [WARN] Solve raised exception: {exc}")
                cpv = np.nan
                gap = np.nan
            dt = time.time() - t0
            C_PV_grid[i, j]   = cpv
            MIPGap_grid[i, j] = gap

            np.save(CACHE_FILE, _checkpoint)

            gap_str   = f"{gap*100:.1f}%" if not np.isnan(gap) else "n/a"
            completed = total - int(np.isnan(C_PV_grid).sum())
            elapsed   = time.time() - t_start
            eta_s     = elapsed / done * (remaining - done) if done > 0 else 0
            print(f"  [{completed:3d}/{total}]  CAPEX={capex_var:5.0f} EUR/kWp  "
                  f"mult={price_mult:.2f}x  ->  C_PV={cpv:5.2f} kWp  "
                  f"gap={gap_str}  ({dt:.0f}s,  ETA {eta_s/60:.1f} min)")

    m.PV_CAPEX_VAR_ANNUAL = m.PV_CAPEX_VAR * m.crf(m.DISCOUNT_RATE, m.LIFETIME_PV)
    print(f"\nSweep complete. Results saved to '{os.path.basename(CACHE_FILE)}'")

    nan_count = np.isnan(C_PV_grid).sum()
    if nan_count:
        print(f"  WARNING: {nan_count} grid points returned NaN — replaced with 0 in plot.")

# ==============================================================================
# PLOT
# ==============================================================================
CAPEX_MESH, PRICE_MESH = np.meshgrid(CAPEX_RANGE, PRICE_RANGE)
C_PV_plot = np.where(np.isnan(C_PV_grid), 0.0, C_PV_grid)

HIGH_GAP_THRESHOLD = SWEEP_MIP_GAP * 1.5
high_gap_mask = np.isnan(MIPGap_grid) | (MIPGap_grid > HIGH_GAP_THRESHOLD)

region = np.zeros_like(C_PV_plot, dtype=int)
region[C_PV_plot >= INSTALL_THRESHOLD]  = 1
region[C_PV_plot >= SATURATE_THRESHOLD] = 2

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Left panel: continuous heatmap ───────────────────────────────────────────
ax = axes[0]
cf = ax.contourf(CAPEX_MESH, PRICE_MESH, C_PV_plot,
                 levels=20, cmap="RdYlGn", alpha=0.92)
cbar = fig.colorbar(cf, ax=ax, label="Optimal PV capacity  (kWp)", pad=0.02)
cbar.ax.axhline(y=INSTALL_THRESHOLD,  color="k", lw=0.9, ls="--")
cbar.ax.axhline(y=SATURATE_THRESHOLD, color="k", lw=0.9, ls="--")

try:
    cs = ax.contour(CAPEX_MESH, PRICE_MESH, C_PV_plot,
                    levels=[INSTALL_THRESHOLD, SATURATE_THRESHOLD],
                    colors="black", linewidths=1.8, linestyles="--")
    ax.clabel(cs,
              fmt={INSTALL_THRESHOLD:  f"{INSTALL_THRESHOLD:.0f} kWp\n(install)",
                   SATURATE_THRESHOLD: f"{SATURATE_THRESHOLD:.0f} kWp\n(saturate)"},
              fontsize=8, inline=True, inline_spacing=6)
except Exception:
    pass

_hatch_gap = ax.pcolormesh(CAPEX_MESH, PRICE_MESH,
                           np.where(high_gap_mask, 1.0, np.nan),
                           cmap=plt.matplotlib.colors.ListedColormap(["none"]),
                           shading="auto", hatch="xxx", edgecolors="dimgray",
                           linewidth=0, alpha=0.0, zorder=5)
ax.plot(BASELINE_CAPEX_VAR, BASELINE_MULT, "k*", markersize=14, zorder=6,
        label=f"Baseline  ({BASELINE_CAPEX_VAR:.0f} EUR/kWp, x{BASELINE_MULT})")
_hatch_patch = mpatches.Patch(facecolor="white", edgecolor="dimgray",
                               hatch="xxx", label=f"High MIP gap (> {HIGH_GAP_THRESHOLD*100:.0f}%)")
ax.legend(handles=[ax.get_legend_handles_labels()[0][0], _hatch_patch],
          labels=[ax.get_legend_handles_labels()[1][0], _hatch_patch.get_label()],
          fontsize=9, loc="lower left", frameon=True)
ax.set_xlabel("PV variable CAPEX  (EUR/kWp)", fontsize=11, fontweight="bold")
ax.set_ylabel("Electricity price multiplier  (-)", fontsize=11, fontweight="bold")
ax.set_title("Optimal PV capacity  (continuous)  [fast mode]", fontsize=12, fontweight="bold")
ax.set_xlim(CAPEX_RANGE[0], CAPEX_RANGE[-1])
ax.set_ylim(PRICE_RANGE[0], PRICE_RANGE[-1])
ax.xaxis.set_major_locator(MultipleLocator(150))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

# ── Right panel: discrete region map ─────────────────────────────────────────
ax2 = axes[1]
colors_reg = ["#d73027", "#fee08b", "#1a9850"]
cmap_d = plt.matplotlib.colors.ListedColormap(colors_reg)
norm_d = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_d.N)

ax2.pcolormesh(CAPEX_MESH, PRICE_MESH, region,
               cmap=cmap_d, norm=norm_d, shading="auto", alpha=0.85)

try:
    cs2 = ax2.contour(CAPEX_MESH, PRICE_MESH, C_PV_plot,
                      levels=[INSTALL_THRESHOLD, SATURATE_THRESHOLD],
                      colors="black", linewidths=2.0, linestyles="--")
    ax2.clabel(cs2,
               fmt={INSTALL_THRESHOLD:  "Install\nthreshold",
                    SATURATE_THRESHOLD: "Saturation\nthreshold"},
               fontsize=8, inline=True, inline_spacing=6)
except Exception:
    pass

ax2.text(780, 0.52, "Not\ninstalled",      ha="center", va="center",
         fontsize=10, fontweight="bold", color="white")
ax2.text(525, 1.20, "Partially\ninstalled", ha="center", va="center",
         fontsize=10, fontweight="bold", color="#333333")
ax2.text(250, 1.80, "Saturated\n(50 kWp)", ha="center", va="center",
         fontsize=10, fontweight="bold", color="white")

ax2.pcolormesh(CAPEX_MESH, PRICE_MESH,
               np.where(high_gap_mask, 1.0, np.nan),
               cmap=plt.matplotlib.colors.ListedColormap(["none"]),
               shading="auto", hatch="xxx", edgecolors="dimgray",
               linewidth=0, alpha=0.0, zorder=5)
ax2.plot(BASELINE_CAPEX_VAR, BASELINE_MULT, "k*", markersize=14, zorder=6)
patches = [mpatches.Patch(color=colors_reg[i], label=lbl)
           for i, lbl in enumerate([
               f"Not installed  (< {INSTALL_THRESHOLD:.0f} kWp)",
               f"Partial  ({INSTALL_THRESHOLD:.0f} - {SATURATE_THRESHOLD:.0f} kWp)",
               f"Saturated  (>= {SATURATE_THRESHOLD:.0f} kWp)",
           ])]
_hatch_patch2 = mpatches.Patch(facecolor="white", edgecolor="dimgray",
                                hatch="xxx", label=f"High MIP gap (> {HIGH_GAP_THRESHOLD*100:.0f}%)")
ax2.legend(handles=patches + [_hatch_patch2], fontsize=8.5, loc="lower left", frameon=True,
           title="Optimal PV regime", title_fontsize=9)
ax2.set_xlabel("PV variable CAPEX  (EUR/kWp)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Electricity price multiplier  (-)", fontsize=11, fontweight="bold")
ax2.set_title("Optimal PV regime  (discrete regions)  [fast mode]", fontsize=12, fontweight="bold")
ax2.set_xlim(CAPEX_RANGE[0], CAPEX_RANGE[-1])
ax2.set_ylim(PRICE_RANGE[0], PRICE_RANGE[-1])
ax2.xaxis.set_major_locator(MultipleLocator(150))
ax2.yaxis.set_major_locator(MultipleLocator(0.2))

suptitle = (f"PV Breakeven Map (fast mode)  |  CAPEX vs Electricity Price  "
            f"[weather {m.WEATHER_YEAR}, prices {m.PRICE_YEAR}]")
if FROM_CACHE:
    suptitle += "   [CACHED — delete breakeven_map_results_fast.npy to rerun]"
fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.02,
             color="#8B0000" if FROM_CACHE else "black")

plt.tight_layout()
out = os.path.join(FOLDER, "breakeven_map_pv_fast.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved to {out}")
