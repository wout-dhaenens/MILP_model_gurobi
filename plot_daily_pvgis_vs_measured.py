"""
plot_daily_pvgis_vs_measured.py
───────────────────────────────
Plots one random sunny-ish day per month, comparing:
  - Measured 15-min PV production (kWh per 15 min → converted to kW)
  - PVGIS hourly simulation scaled to installed capacity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random

plt.style.use('seaborn-v0_8-whitegrid')
random.seed(42)

# ==============================================================================
# CONFIG
# ==============================================================================
INSTALLED_KWP = 34.63

PVGIS_CSV = r"C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\python\electri toy MILP\pvgis_data.csv"
MEASURED_CSV = (
    r"C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\python\electri toy MILP"
    r"\wetransfer_historiek_submeting_elektriciteit_541454897100022743_20250101_20260101"
    r"_kwartiertotalen-csv_2026-03-26_1550"
    r"\Historiek_submeting_elektriciteit_541454897100022743_20230101_20240101_kwartiertotalen.csv"
)

# ==============================================================================
# LOAD PVGIS  (hourly, already in Brussels time)
# ==============================================================================
df_pvgis = pd.read_csv(PVGIS_CSV, parse_dates=['time'])
df_pvgis = df_pvgis.set_index('time')
pvgis_kw = df_pvgis['pv_kw_per_kwp'] * INSTALLED_KWP   # kW (= kWh/h)

# ==============================================================================
# LOAD MEASURED  (15-min, kWh per interval → convert to kW)
# ==============================================================================
df_raw = pd.read_csv(MEASURED_CSV, sep=';', decimal=',', encoding='utf-8-sig')
df_pv  = df_raw[df_raw['Register'] == 'Productie Actief'].copy()
df_pv['timestamp'] = pd.to_datetime(
    df_pv['Van (datum)'] + ' ' + df_pv['Van (tijdstip)'], dayfirst=True)
df_pv = df_pv.set_index('timestamp').sort_index()
df_pv['Volume'] = pd.to_numeric(df_pv['Volume'], errors='coerce')
# kWh per 15 min → kW  (multiply by 4)
measured_15min_kw = df_pv['Volume'] * 4.0

# ==============================================================================
# PICK ONE RANDOM DAY PER MONTH
# (prefer days where measured peak > 30% of installed capacity — avoids cloudy days)
# ==============================================================================
months = range(1, 13)
selected_days = []

for m in months:
    # Get all days in this month that have measured data
    month_data = measured_15min_kw[measured_15min_kw.index.month == m]
    daily_peak = month_data.resample('D').max()
    # Prefer days with decent production (> 20% of installed)
    good_days = daily_peak[daily_peak > 0.20 * INSTALLED_KWP].index
    if len(good_days) == 0:
        good_days = daily_peak.index   # fallback: any day
    selected_days.append(random.choice(good_days.tolist()))

selected_days.sort()

# ==============================================================================
# PLOT — 4 rows × 3 cols, one panel per month
# ==============================================================================
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()

for ax, day in zip(axes, selected_days):
    day_str = day.strftime('%Y-%m-%d')

    # --- PVGIS for this day ---
    pvgis_day = pvgis_kw[pvgis_kw.index.date == day.date()]

    # --- Measured 15-min for this day ---
    meas_day = measured_15min_kw[measured_15min_kw.index.date == day.date()]

    # Plot measured as step line (15-min resolution)
    ax.step(meas_day.index, meas_day.values, where='post',
            color='#1f77b4', lw=1.2, alpha=0.85, label='Measured (15-min)')

    # Plot PVGIS as step line (hourly)
    ax.step(pvgis_day.index, pvgis_day.values, where='post',
            color='#ff7f0e', lw=2.0, label='PVGIS (hourly)')

    ax.set_title(day.strftime('%d %b %Y'), fontsize=11, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Power (kW)', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=8)

    # Annotate daily totals
    meas_total = meas_day.sum() / 4.0   # kW × 15min intervals → kWh
    pvgis_total = pvgis_day.sum()        # kWh/h × 1h = kWh
    ax.text(0.02, 0.95,
            f'Measured: {meas_total:.1f} kWh\nPVGIS:    {pvgis_total:.1f} kWh',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=11,
           frameon=True, bbox_to_anchor=(0.5, 0.01))

fig.suptitle(f'Daily PV comparison: PVGIS vs Measured  —  {INSTALLED_KWP} kWp system (2023)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(
    r"C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\python\electri toy MILP\daily_pvgis_vs_measured.png",
    dpi=150, bbox_inches='tight'
)
plt.show()
print("Done.")
