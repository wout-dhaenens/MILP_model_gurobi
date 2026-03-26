import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import math
from Milp_yearly_test_gurobi import generate_capex_pwl

# ==============================================================================
# --- CONFIG — must match your MILP script ---
# ==============================================================================
YEAR   = 2023
T      = 8760

# LTES geometry (kept for reference)
rho_pcm      = 860.0
L_pcm        = 199.0
rho_L_pcm    = rho_pcm * L_pcm / 3600.0
d_ltes       = 1.0
A_cross      = math.pi * (d_ltes / 2) ** 2
kWh_per_m_ltes = A_cross * rho_L_pcm

# WTES geometry (Needed to reconstruct the WTES cost curve)
delta_T_HC = 70.0 - 50.0
rho_water = 971.8
c_water = 4190.0
A_cross_wtes = math.pi * (1.0 / 2) ** 2
kWh_per_m_wtes = (A_cross_wtes * rho_water * c_water * delta_T_HC) / 3.6e6

# ==============================================================================
# --- LOAD SAVED RESULTS ---
# ==============================================================================
data = np.load('milp_results.npz')
res  = {k: data[k] for k in data.files}   # all arrays

# Split out the non-result arrays that were saved alongside
P_THERMAL_LOAD = res.pop('P_THERMAL_LOAD')
P_LOAD         = res.pop('P_LOAD')
COP_t          = res.pop('COP_t')
P_price_buy    = res.pop('P_price_buy')
P_price_sell   = res.pop('P_price_sell')

with open('milp_opt.json') as f:
    opt = json.load(f)

print(f"Loaded results: PV={opt['C_PV']:.1f} kWp | "
      f"Bat={opt['C_bat']:.1f} kWh | HP={opt['C_HP']:.1f} kW_th | "
      f"LTES={opt['C_TES']:.1f} kWh_th")

# ==============================================================================
# --- PLOT FUNCTIONS ---
# ==============================================================================




def _plot_pwl_capex(opt):
    """
    Reconstructs the Piecewise Linear CAPEX curves and plots the optimized 
    solution point to visualize economies of scale.
    (Plots straight line segments between the specific cost breakpoints)
    """
    # Re-fetch bounds and parameters to match the main script
    from Milp_yearly_test_gurobi import (C_PV_MAX, CAPEX_PV_ANNUAL,
                                         C_BAT_MAX, CAPEX_BAT_ANNUAL,
                                         C_HP_MAX, CAPEX_HP_ANNUAL,
                                         C_LTES_MAX, CAPEX_TES_eu_kWh_LTES,
                                         h_wtes_max, CAPEX_TES_eu_kWh_WTES,
                                         WTES_REF_LITRES, kWh_per_m_wtes)

    from Milp_yearly_test_gurobi import A_cross_wtes
    # anchor in plot units (metres): 500 L = 0.5 m³, divide by cross-section area
    wtes_anchor_m = (WTES_REF_LITRES / 1000.0) / A_cross_wtes

    configs = [
        ("PV Capacity",       "kWp",    opt['C_PV'],   10.0,           *generate_capex_pwl(C_PV_MAX,   CAPEX_PV_ANNUAL,          anchor_cap=10.0)),
        ("Battery Capacity",  "kWh",    opt['C_bat'],  10.0,           *generate_capex_pwl(C_BAT_MAX,  CAPEX_BAT_ANNUAL,         anchor_cap=10.0)),
        ("Heat Pump Capacity","kW_th",  opt['C_HP'],   30.0,           *generate_capex_pwl(C_HP_MAX,   CAPEX_HP_ANNUAL,          anchor_cap=30.0)),
        ("LTES Capacity",     "kWh_th", opt['C_ltes'], 30.0,           *generate_capex_pwl(C_LTES_MAX, CAPEX_TES_eu_kWh_LTES,   anchor_cap=30.0)),
        ("WTES Height",       "m",      opt['h_wtes'], wtes_anchor_m,  *generate_capex_pwl(h_wtes_max, CAPEX_TES_eu_kWh_WTES,   anchor_cap=WTES_REF_LITRES, is_wtes=True)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle("Economies of Scale: Piecewise Linear Segments & Optimized Sizing", fontsize=16, fontweight='bold', y=0.98)

    for i, (title, unit, opt_val, anchor_x, bp_x, bp_y) in enumerate(configs):
        ax = axes[i]

        # Build the true unit-cost curve: within each PWL segment total cost is
        # linear, so unit cost = total(x)/x is hyperbolic — NOT a straight line.
        # Sample densely inside every segment so the curve is plotted correctly.
        curve_x, curve_y = [], []
        for seg in range(len(bp_x) - 1):
            x0, x1 = bp_x[seg], bp_x[seg + 1]
            y0, y1 = bp_y[seg], bp_y[seg + 1]
            x_start = max(x0, 0.1)   # avoid division by zero at origin
            xs = np.linspace(x_start, x1, 60)
            ys_total = y0 + (y1 - y0) / (x1 - x0) * (xs - x0)
            curve_x.extend(xs)
            curve_y.extend(ys_total / xs)

        # Mark the actual PWL breakpoints (but don't connect their unit costs directly)
        bp_x_safe = np.array(bp_x); bp_x_safe[0] = 0.1
        bp_unit_at_bps = np.array(bp_y) / bp_x_safe

        ax.plot(curve_x, curve_y, color='steelblue', lw=2, label="PWL (unit cost)")
        ax.plot(bp_x_safe[1:], bp_unit_at_bps[1:], 'o', color='steelblue',
                markersize=6, label="Breakpoints")

        # If the solver actually built this technology, plot it on the curve
        if opt_val > 0.1: 
            opt_total_cost = np.interp(opt_val, bp_x, bp_y)
            opt_unit_cost = opt_total_cost / opt_val
            
            ax.plot(opt_val, opt_unit_cost, color='tomato', marker='o', markersize=10, 
                    label=f"Solution:\n{opt_val:.1f} {unit}\n€{opt_unit_cost:,.2f}/{unit}/yr", zorder=5)
        else:
            ax.plot([], [], ' ', label=f"Solution: Not Built (0 {unit})")

        # Formatting
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"Installed Capacity [{unit}]", fontsize=10)
        ax.set_ylabel(f"Annualized Unit Cost [€/{unit}/yr]", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0) # Force y-axis to start at 0
        ax.legend(fontsize=9)
        
        # Highlight the anchor point (where base_rate is exactly true)
        ax.axvline(anchor_x, color='gray', linestyle=':', alpha=0.5)
        y_min, y_max = ax.get_ylim()
        ax.text(anchor_x * 1.05, y_max * 0.1, 'Base Rate\nAnchor', color='gray', fontsize=8)

    # Turn off the empty 6th subplot
    axes[-1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('pwl_capex_unit_cost.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved: pwl_capex_unit_cost.png")

def check_hp_minimum_load(opt, res, min_frac=0.30, tol=1e-4):
    """
    Checks if the HP respects the minimum part-load constraint when turned on.
    Uses a small tolerance to ignore solver floating-point noise.
    """
    c_hp = opt['C_HP']
    q_hp_th = res['Q_hp_th']
    
    if c_hp < tol:
        print("\n--- HP Minimum Load Check ---")
        print("HP capacity is ~0. No part-load to check.")
        return True

    min_required = c_hp * min_frac
    on_mask = q_hp_th > tol
    hours_on = np.sum(on_mask)
    
    violations = q_hp_th[on_mask] < (min_required - tol)
    num_violations = np.sum(violations)
    
    print(f"\n--- Heat Pump Minimum Load Check ---")
    print(f"  Sized Capacity:        {c_hp:.2f} kW_th")
    print(f"  Required {min_frac*100:.0f}% Min:    {min_required:.2f} kW_th")
    print(f"  Total hours HP is ON: {hours_on}")
    
    if num_violations == 0:
        print("  Status: SUCCESS! HP always operates at >= 30% when ON.")
        return True
    else:
        print(f"  Status: FAILED! HP drops below 30% for {num_violations} hours.")
        violating_indices = np.where(on_mask)[0][violations]
        print("  Sample of violations (Hour: Output):")
        for idx in violating_indices[:5]:
            print(f"    Hour {idx:>4}: {q_hp_th[idx]:.3f} kW_th")
        return False



def _plot_annual(opt, res, P_price_buy, P_price_sell, COP_t):
    fig, axes = plt.subplots(5, 1, figsize=(16, 24), sharex=True)
    title = (f"Annual Optimisation Results — LTES  ({YEAR})\n"
             f"PV {opt['C_PV']:.1f} kWp | Bat {opt['C_bat']:.1f} kWh | "
             f"HP {opt['C_HP']:.1f} kW_th | LTES {opt['C_ltes']:.1f} kWh_th "
             f"(V={opt['V_ltes']:.2f} m³ | {opt['mass_pcm']:.0f} kg PCM)")
    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.995)

    dates_hourly = pd.date_range(f"{YEAR}-01-01", periods=T, freq='h')

    # --- Panel 1: Electrical power flows ---
    ax = axes[0]
    ax.step(dates_hourly, res['PV_prod'],   label='PV Generation',  color='orange', lw=1.2, where='post')
    ax.step(dates_hourly, P_LOAD,           label='Elec. Load',     color='black',  lw=1.2, ls='--', where='post')
    ax.step(dates_hourly, res['P_hp_elec'], label='HP Elec. Input', color='red',    lw=1.2, where='post')
    ax.fill_between(dates_hourly, res['P_buy'],  0, alpha=0.3, color='steelblue', label='Grid Buy', step='post')
    ax.fill_between(dates_hourly, res['P_sell'], 0, alpha=0.3, color='gold',      label='Grid Sell', step='post')
    ax.set_ylabel('Power [kW_e]', fontsize=11)
    ax.legend(fontsize=9, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Electrical Power Flows (Hourly Data)', fontsize=11)

    # --- Panel 2: Thermal — HP output split + LTES SoC ---
    ax = axes[1]
    
    y1 = res['Q_load_in']
    y2 = y1 + res['Q_tes_out']
    ax.fill_between(dates_hourly, 0, y1, step='post', color='tomato', alpha=0.55, label='HP → Load (direct)')
    ax.fill_between(dates_hourly, y1, y2, step='post', color='steelblue', alpha=0.55, label='TES → Load (discharge)')
    
    ax.step(dates_hourly, P_THERMAL_LOAD,  label='Thermal Demand',      color='black',      lw=1.2, ls='--', where='post')
    ax.step(dates_hourly, res['Q_tes_in'], label='HP → TES (charging)', color='darkorange', lw=1.2, ls='-.', where='post')

    ax2 = ax.twinx()
    ax2.step(dates_hourly, res['Q_tes'], label='LTES SoC [kWh]', color='navy', lw=2, where='post')
    ax2.axhline(0.95 * opt['C_TES'], color='navy', ls=':', alpha=0.5, label='LTES 95%')
    ax2.axhline(0.05 * opt['C_TES'], color='navy', ls=':', alpha=0.3, label='LTES 5%')
    ax2.set_ylabel('Stored Energy [kWh_th]', color='navy', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='navy')
    ax.set_ylabel('Power [kW_th]', fontsize=11)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title('Thermal System — HP output split & LTES SoC (Hourly Data)', fontsize=11)

    # --- Panel 3: Battery SoC & COP ---
    ax = axes[2]
    ax.step(dates_hourly, res['SoC_bat'], label='Battery SoC', color='green', lw=1.5, where='post')
    ax.axhline(opt['C_bat'], color='green', ls='--', alpha=0.5,
               label=f"Bat capacity {opt['C_bat']:.1f} kWh")
    ax3 = ax.twinx()
    ax3.step(dates_hourly, COP_t, label='Lorenz COP', color='purple', lw=1.5, ls='-.', where='post')
    ax3.set_ylabel('COP [-]', color='purple', fontsize=11)
    ax3.tick_params(axis='y', labelcolor='purple')
    ax.set_ylabel('Energy [kWh]', fontsize=11)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Battery SoC & Heat Pump COP (Hourly Data)', fontsize=11)

    # --- Panel 4: Monthly peak grid import ---
    ax = axes[3]
    months_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    month_starts  = pd.date_range(f"{YEAR}-01-01", periods=12, freq='MS')
    ax.bar(month_starts, opt['P_peak_m'], width=20,
           color='steelblue', alpha=0.7, align='edge', label='Monthly peak P_buy [kW]')
    ax.set_ylabel('Peak Grid Import [kW]', fontsize=11)
    ax.set_xticks(month_starts)
    ax.set_xticklabels(months_labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Monthly Peak Grid-Import Power (capacity tariff basis)', fontsize=11)

    # --- Panel 5: Electricity prices ---
    ax = axes[4]
    ax.step(dates_hourly, P_price_buy,  label='Buy price [€/kWh]',  color='purple', lw=1.5, where='post')
    ax.step(dates_hourly, P_price_sell, label='Sell price [€/kWh]', color='gray',   lw=1,   ls='--', where='post')
    ax.set_ylabel('Price [€/kWh]', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Electricity Prices (Hourly Data)', fontsize=11)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('annual_optimization_results_LTES.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved: annual_optimization_results_LTES.png")


def _plot_representative_day(opt, res, P_price_buy, P_price_sell, COP_t,
                             day_of_year, season_label, filename):
    h0    = day_of_year * 24
    sl    = slice(h0, h0 + 24)
    hours = np.arange(24)
    date_str = (pd.Timestamp(f"{YEAR}-01-01") + pd.Timedelta(days=day_of_year)).strftime("%d %b %Y")

    fig, axes = plt.subplots(4, 1, figsize=(13, 18), sharex=True)
    title = (f"Representative {season_label} Day — {date_str}  [TES]\n"
             f"PV {opt['C_PV']:.1f} kWp | Bat {opt['C_bat']:.1f} kWh | "
             f"HP {opt['C_HP']:.1f} kW_th | TES {opt['C_TES']:.1f} kWh_th "
             f"({opt['V_ltes']*1000:.0f} L PCM)")
    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.995)

    # --- Panel 1: Electrical power flows ---
    ax = axes[0]
    ax.step(hours, res['PV_prod'][sl],   label='PV Generation',  color='orange', lw=2, where='post')
    ax.step(hours, P_LOAD[sl],           label='Elec. Load',     color='black',  lw=2, ls='--', where='post')
    ax.step(hours, res['P_hp_elec'][sl], label='HP Elec. Input', color='red',    lw=2, where='post')
    ax.fill_between(hours, res['P_buy'][sl],  0, alpha=0.35, color='steelblue', label='Grid Buy', step='post')
    ax.fill_between(hours, res['P_sell'][sl], 0, alpha=0.35, color='gold',      label='Grid Sell', step='post')
    ax.set_ylabel('Power [kW_e]', fontsize=11)
    ax.legend(fontsize=9, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Electrical Power Flows', fontsize=11)

    # --- Panel 2: Thermal — HP output split + LTES SoC ---
    ax = axes[1]
    
    y1 = res['Q_load_in'][sl]
    y2 = y1 + res['Q_tes_out'][sl]
    ax.fill_between(hours, 0, y1, step='post', color='tomato', alpha=0.55, label='HP → Load (direct)')
    ax.fill_between(hours, y1, y2, step='post', color='steelblue', alpha=0.55, label='TES → Load (discharge)')
    
    ax.step(hours, P_THERMAL_LOAD[sl],   label='Thermal Demand',      color='black',      lw=2, ls='--', where='post')
    ax.step(hours, res['Q_tes_in'][sl],  label='HP → TES (charging)', color='darkorange', lw=2, ls='-.', where='post')

    ax2 = ax.twinx()
    ax2.step(hours, res['Q_tes'][sl],    label='LTES SoC [kWh]', color='navy', lw=2, where='post')
    ax2.axhline(0.95 * opt['C_TES'], color='navy', ls=':', alpha=0.5, label='LTES 95%')
    ax2.axhline(0.05 * opt['C_TES'], color='navy', ls=':', alpha=0.3, label='LTES 5%')
    ax2.set_ylabel('Stored Energy [kWh_th]', color='navy', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='navy')
    ax.set_ylabel('Power [kW_th]', fontsize=11)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title('Thermal System — HP output split & LTES SoC', fontsize=11)

    # --- Panel 3: Battery SoC, charge/discharge & COP ---
    ax = axes[2]
    ax.step(hours, res['SoC_bat'][sl],   label='Battery SoC',      color='green',     lw=2, where='post')
    ax.fill_between(hours, res['P_bat_ch'][sl],  0, alpha=0.25, color='green',      label='Bat Charge', step='post')
    ax.fill_between(hours, res['P_bat_dis'][sl], 0, alpha=0.25, color='limegreen', label='Bat Discharge', step='post')
    ax.axhline(opt['C_bat'], color='green', ls='--', alpha=0.5, label=f"Capacity {opt['C_bat']:.1f} kWh")
    ax3 = ax.twinx()
    ax3.step(hours, COP_t[sl], label='Lorenz COP', color='purple', lw=2, ls='-.', where='post')
    ax3.set_ylabel('COP [-]', color='purple', fontsize=11)
    ax3.tick_params(axis='y', labelcolor='purple')
    ax.set_ylabel('Energy / Power [kWh or kW]', fontsize=11)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Battery SoC, Charge/Discharge & COP', fontsize=11)

    # --- Panel 4: Electricity prices ---
    ax = axes[3]
    ax.step(hours, P_price_buy[sl],  label='Buy price [€/kWh]',  color='purple', lw=2, where='post')
    ax.step(hours, P_price_sell[sl], label='Sell price [€/kWh]', color='gray',   lw=2, ls='--', where='post')
    ax.set_ylabel('Price [€/kWh]', fontsize=11)
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_xticks(hours)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Electricity Prices', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: {filename}")


def _plot_january_detail(opt, res, P_price_buy, P_price_sell, COP_t):
    jan_hours = slice(0, 744)
    jan_dates = pd.date_range(f"{YEAR}-01-01", periods=744, freq='h')

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    title = (f"January {YEAR} — Hourly Detail  [LTES]\n"
             f"PV {opt['C_PV']:.1f} kWp | Bat {opt['C_bat']:.1f} kWh | "
             f"HP {opt['C_HP']:.1f} kW_th | TES {opt['C_TES']:.1f} kWh_th "
             f"({opt['V_ltes']*1000:.0f} L PCM)")
    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    # --- Panel 1: Electrical power flows ---
    ax = axes[0]
    ax.step(jan_dates, res['PV_prod'][jan_hours],   label='PV Generation',  color='orange',    lw=1.2, where='post')
    ax.step(jan_dates, P_LOAD[jan_hours],           label='Elec. Load',     color='black',     lw=1.2, ls='--', where='post')
    ax.step(jan_dates, res['P_hp_elec'][jan_hours], label='HP Elec. Input', color='red',       lw=1.2, where='post')
    ax.fill_between(jan_dates, res['P_buy'][jan_hours],  0, alpha=0.35, color='steelblue', label='Grid Buy', step='post')
    ax.fill_between(jan_dates, res['P_sell'][jan_hours], 0, alpha=0.35, color='gold',      label='Grid Sell', step='post')
    ax.set_ylabel('Power [kW_e]', fontsize=11)
    ax.legend(fontsize=9, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Electrical Power Flows — January (hourly)', fontsize=11)

    # --- Panel 2: Thermal — HP output split + LTES SoC ---
    ax = axes[1]
    
    y1 = res['Q_load_in'][jan_hours]
    y2 = y1 + res['Q_tes_out'][jan_hours]
    ax.fill_between(jan_dates, 0, y1, step='post', color='tomato', alpha=0.55, label='HP → Load (direct)')
    ax.fill_between(jan_dates, y1, y2, step='post', color='steelblue', alpha=0.55, label='TES → Load (discharge)')
    
    ax.step(jan_dates, P_THERMAL_LOAD[jan_hours], label='Thermal Demand',      color='black',      lw=1.5, ls='--', where='post')
    ax.step(jan_dates, res['Q_tes_in'][jan_hours], label='HP → TES (charging)', color='darkorange', lw=1.2, ls='-.', where='post')

    ax2 = ax.twinx()
    ax2.step(jan_dates, res['Q_tes'][jan_hours], label='TES SoC [kWh]', color='navy', lw=2, where='post')
    ax2.axhline(0.95 * opt['C_TES'], color='navy', ls=':', alpha=0.5, label='TES 95%')
    ax2.axhline(0.05 * opt['C_TES'], color='navy', ls=':', alpha=0.3, label='TES 5%')
    ax2.set_ylabel('Stored Energy [kWh_th]', color='navy', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='navy')
    ax.set_ylabel('Power [kW_th]', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title('Thermal System — HP output split & LTES SoC — January (hourly)', fontsize=11)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig('january_detail_LTES.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved: january_detail_LTES.png")


def _plot_peak_shaving(opt, res):
    """
    Visualises how the LTES allows the HP to be undersized relative to
    peak thermal demand — i.e. the peak-shaving role of the TES.

    Panel 1: Load duration curves — thermal demand vs HP output (sorted descending)
    Panel 2: Scatter — HP thermal output vs thermal demand, coloured by Q_tes_out
    """
    Q_hp_th   = res['Q_hp_th']           
    Q_tes_out = res['Q_tes_out']          
    Q_load_in = res['Q_load_in']          
    demand    = P_THERMAL_LOAD            

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle(
        f"HP Peak Shaving by LTES — HP sized at {opt['C_HP']:.1f} kW_th  |  "
        f"Peak demand: {demand.max():.1f} kW_th  |  "
        f"LTES capacity: {opt['C_TES']:.1f} kWh_th",
        fontsize=12, fontweight='bold', y=1.02
    )

    # ------------------------------------------------------------------
    # Panel 1: Load duration curves
    # ------------------------------------------------------------------
    ax = axes[0]

    demand_sorted  = np.sort(demand)[::-1]
    hp_out_sorted  = np.sort(Q_hp_th)[::-1]
    hours_pct      = np.arange(1, T + 1) / T * 100   

    ax.plot(hours_pct, demand_sorted, color='black', lw=2, label='Thermal demand')
    ax.plot(hours_pct, hp_out_sorted, color='tomato', lw=2, label='HP thermal output')
    ax.axhline(opt['C_HP'], color='tomato', lw=1.5, ls='--', alpha=0.7, label=f"HP capacity ({opt['C_HP']:.1f} kW_th)")
    ax.axhline(demand.max(), color='black', lw=1, ls=':', alpha=0.5, label=f"Peak demand ({demand.max():.1f} kW_th)")

    ax.fill_between(hours_pct, hp_out_sorted, demand_sorted,
                    where=(demand_sorted > hp_out_sorted),
                    alpha=0.25, color='steelblue', label='Gap covered by LTES')

    ax.set_xlabel('% of year (hours sorted by demand)', fontsize=11)
    ax.set_ylabel('Thermal Power [kW_th]', fontsize=11)
    ax.set_title('Load Duration Curve', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 2: Scatter — HP output vs demand, coloured by TES discharge
    # ------------------------------------------------------------------
    ax = axes[1]

    sc = ax.scatter(demand, Q_hp_th, c=Q_tes_out, cmap='Blues', s=2, alpha=0.6, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Q_tes_out [kW_th]  (TES discharge)', fontsize=10)

    max_val = max(demand.max(), Q_hp_th.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], color='gray', lw=1.5, ls='--', label='HP = demand (no TES)')
    ax.axhline(opt['C_HP'], color='tomato', lw=1.5, ls='--', alpha=0.8, label=f"HP capacity ({opt['C_HP']:.1f} kW_th)")

    ax.set_xlabel('Thermal Demand [kW_th]', fontsize=11)
    ax.set_ylabel('HP Thermal Output [kW_th]', fontsize=11)
    ax.set_title('HP Output vs Demand\n(colour = TES discharge)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('peak_shaving_LTES.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved: peak_shaving_LTES.png")

# ==============================================================================
# --- RUN WHICHEVER PLOTS YOU WANT ---
# ==============================================================================
# _plot_annual(opt, res, P_price_buy, P_price_sell, COP_t)
# _plot_representative_day(opt, res, P_price_buy, P_price_sell, COP_t, 171, 'Summer', 'summer.png')
# _plot_representative_day(opt, res, P_price_buy, P_price_sell, COP_t, 354, 'Winter', 'winter.png')
# _plot_january_detail(opt, res, P_price_buy, P_price_sell, COP_t)
# _plot_peak_shaving(opt, res)
# check_hp_minimum_load(opt, res)
_plot_pwl_capex(opt)