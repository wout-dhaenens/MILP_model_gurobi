"""
compare_pvgis_vs_measured.py
────────────────────────────
Compares PVGIS 2023 hourly PV simulation data against the measured
15-minute sub-meter data (Productie Actief, EAN 541454897100022743).

Adjust INSTALLED_KWP to match the actual installed PV capacity on site.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-v0_8-whitegrid')

# ==============================================================================
# --- CONFIGURATION ------------------------------------------------------------
# ==============================================================================
INSTALLED_KWP = 34.63   # kWp

PVGIS_CSV = r"C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\python\electri toy MILP\pvgis_data.csv"

MEASURED_CSV = (
    r"C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\python\electri toy MILP"
    r"\wetransfer_historiek_submeting_elektriciteit_541454897100022743_20250101_20260101"
    r"_kwartiertotalen-csv_2026-03-26_1550"
    r"\Historiek_submeting_elektriciteit_541454897100022743_20230101_20240101_kwartiertotalen.csv"
)

# ==============================================================================
# --- 1. LOAD PVGIS DATA -------------------------------------------------------
# ==============================================================================
df_pvgis = pd.read_csv(PVGIS_CSV, parse_dates=['time'])

# PVGIS timestamps carry a :10-min offset — floor to the hour so they align
# with the measured data after resampling
df_pvgis['time'] = df_pvgis['time'].dt.floor('h')
df_pvgis = df_pvgis.set_index('time')

# Scale normalised profile to actual system: kW/kWp × kWp = kW = kWh/h (hourly)
df_pvgis['pv_kwh'] = df_pvgis['pv_kw_per_kwp'] * INSTALLED_KWP

pvgis_hourly = df_pvgis['pv_kwh']   # kWh per hour, indexed by hour

# ==============================================================================
# --- 2. LOAD & PROCESS MEASURED DATA ------------------------------------------
# ==============================================================================
df_raw = pd.read_csv(
    MEASURED_CSV,
    sep=';',
    decimal=',',
    encoding='utf-8-sig',
)

# Keep only active PV production register
df_pv = df_raw[df_raw['Register'] == 'Productie Actief'].copy()

# Build a proper timestamp from "Van (datum)" + "Van (tijdstip)"
df_pv['timestamp'] = pd.to_datetime(
    df_pv['Van (datum)'] + ' ' + df_pv['Van (tijdstip)'],
    dayfirst=True,
)
df_pv = df_pv.set_index('timestamp').sort_index()
df_pv['Volume'] = pd.to_numeric(df_pv['Volume'], errors='coerce')

# Resample 15-min kWh → hourly kWh by summing
measured_hourly = df_pv['Volume'].resample('h').sum()

# ==============================================================================
# --- 3. ALIGN & MERGE ---------------------------------------------------------
# ==============================================================================
df = pd.DataFrame({
    'pvgis_kwh':    pvgis_hourly,
    'measured_kwh': measured_hourly,
}).dropna()

pvgis    = df['pvgis_kwh']
measured = df['measured_kwh']
diff     = pvgis - measured

# ==============================================================================
# --- 4. STATISTICS ------------------------------------------------------------
# ==============================================================================
rmse     = np.sqrt((diff**2).mean())
mae      = diff.abs().mean()
mbe      = diff.mean()
r2       = np.corrcoef(pvgis, measured)[0, 1] ** 2
total_pvgis    = pvgis.sum()
total_measured = measured.sum()
bias_pct = 100 * (total_pvgis - total_measured) / total_measured

print("=" * 60)
print("   PVGIS 2023  vs  Measured PV Production")
print(f"   Installed capacity : {INSTALLED_KWP:.2f} kWp")
print("=" * 60)
print(f"  Hours compared        : {len(df):,}")
print()
print(f"  Total PVGIS [kWh]     : {total_pvgis:,.0f}")
print(f"  Total measured [kWh]  : {total_measured:,.0f}")
print(f"  Annual bias           : {bias_pct:+.1f}%  (PVGIS - measured)", flush=True)
print()
print(f"  RMSE  [kWh/h]         : {rmse:.3f}")
print(f"  MAE   [kWh/h]         : {mae:.3f}")
print(f"  MBE   [kWh/h]         : {mbe:+.3f}")
print(f"  R²                    : {r2:.4f}")
print("=" * 60)

# Monthly breakdown
monthly = df.resample('ME').sum()
monthly.index = monthly.index.strftime('%b')
print("\n  Monthly totals (kWh):")
print(f"  {'Month':<6} {'PVGIS':>10} {'Measured':>10} {'Bias %':>8}")
print("  " + "-" * 38)
for month, row in monthly.iterrows():
    b = 100 * (row['pvgis_kwh'] - row['measured_kwh']) / row['measured_kwh'] if row['measured_kwh'] else float('nan')
    print(f"  {month:<6} {row['pvgis_kwh']:>10.0f} {row['measured_kwh']:>10.0f} {b:>+7.1f}%")

# ==============================================================================
# --- 5. PLOTS -----------------------------------------------------------------
# ==============================================================================
fig = plt.figure(figsize=(16, 14))
gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30,
               height_ratios=[2.2, 1.5, 1.5])

ax_ts      = fig.add_subplot(gs[0, :])   # full-year time series
ax_monthly = fig.add_subplot(gs[1, 0])   # monthly totals
ax_diff    = fig.add_subplot(gs[1, 1])   # daily difference
ax_scatter = fig.add_subplot(gs[2, 0])   # scatter
ax_diurnal = fig.add_subplot(gs[2, 1])   # average diurnal profile

# ── Panel 1 : Full-year overlay (daily sums for readability) ──────────────────
daily_pvgis    = pvgis.resample('D').sum()
daily_measured = measured.resample('D').sum()

ax_ts.fill_between(daily_pvgis.index, daily_pvgis,
                   color='#ff7f0e', alpha=0.35, label='PVGIS simulated')
ax_ts.fill_between(daily_measured.index, daily_measured,
                   color='#1f77b4', alpha=0.35, label='Measured')
ax_ts.plot(daily_pvgis.index,    daily_pvgis,    color='#ff7f0e', lw=1.4)
ax_ts.plot(daily_measured.index, daily_measured, color='#1f77b4', lw=1.4)
ax_ts.set_ylabel('Daily PV energy (kWh)', fontsize=11, fontweight='bold')
ax_ts.set_title(
    f'PVGIS 2023 vs Measured PV Production  —  {INSTALLED_KWP:.2f} kWp system',
    fontsize=13, fontweight='bold', pad=10)
ax_ts.legend(fontsize=10, frameon=True)
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax_ts.xaxis.set_major_locator(mdates.MonthLocator())

# ── Panel 2 : Monthly totals bar chart ───────────────────────────────────────
x      = np.arange(len(monthly))
width  = 0.38
bars1  = ax_monthly.bar(x - width/2, monthly['pvgis_kwh'],    width, label='PVGIS',    color='#ff7f0e', alpha=0.8)
bars2  = ax_monthly.bar(x + width/2, monthly['measured_kwh'], width, label='Measured', color='#1f77b4', alpha=0.8)
ax_monthly.set_xticks(x)
ax_monthly.set_xticklabels(monthly.index, fontsize=9)
ax_monthly.set_ylabel('Monthly energy (kWh)', fontsize=10, fontweight='bold')
ax_monthly.set_title('Monthly Totals', fontsize=11, fontweight='bold')
ax_monthly.legend(fontsize=9, frameon=True)

# ── Panel 3 : Daily bias ─────────────────────────────────────────────────────
daily_diff = daily_pvgis - daily_measured
colors = ['#ff7f0e' if v >= 0 else '#1f77b4' for v in daily_diff]
ax_diff.bar(daily_diff.index, daily_diff.values, width=1.0, color=colors, alpha=0.7)
ax_diff.axhline(0, color='black', lw=0.8, ls='--')
ax_diff.set_ylabel('PVGIS − Measured (kWh/day)', fontsize=10, fontweight='bold')
ax_diff.set_title('Daily Bias  (orange = PVGIS overestimates)', fontsize=11, fontweight='bold')
ax_diff.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax_diff.xaxis.set_major_locator(mdates.MonthLocator())

# ── Panel 4 : Scatter ─────────────────────────────────────────────────────────
# Only daytime hours (where either series > 0.01 kWh)
mask_day = (pvgis > 0.01) | (measured > 0.01)
x_sc = pvgis[mask_day].values
y_sc = measured[mask_day].values

ax_scatter.scatter(x_sc, y_sc, alpha=0.20, s=8, color='#9467bd', edgecolors='none')
lim = max(x_sc.max(), y_sc.max()) * 1.05
ax_scatter.plot([0, lim], [0, lim], 'k--', lw=1.5, label='1:1 line')
m, b_fit = np.polyfit(x_sc, y_sc, 1)
xfit = np.linspace(0, lim, 200)
ax_scatter.plot(xfit, m * xfit + b_fit, color='#d62728', lw=1.8,
                label=f'Fit: y = {m:.2f}x + {b_fit:.2f}')
ax_scatter.set_xlabel('PVGIS (kWh/h)', fontsize=10, fontweight='bold')
ax_scatter.set_ylabel('Measured (kWh/h)', fontsize=10, fontweight='bold')
ax_scatter.set_title('Hourly Scatter (daytime only)', fontsize=11, fontweight='bold')
ax_scatter.set_xlim(0, lim); ax_scatter.set_ylim(0, lim)
ax_scatter.legend(fontsize=9, frameon=True)
ax_scatter.text(0.05, 0.92, f'R² = {r2:.3f}', transform=ax_scatter.transAxes,
                fontsize=10, fontweight='bold')

# ── Panel 5 : Average diurnal profile by season ───────────────────────────────
df['hour']  = df.index.hour
df['month'] = df.index.month

seasons = {
    'Winter (Dec–Feb)': [12, 1, 2],
    'Spring (Mar–May)': [3, 4, 5],
    'Summer (Jun–Aug)': [6, 7, 8],
    'Autumn (Sep–Nov)': [9, 10, 11],
}
colors_s = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
ls_pvgis  = '-'
ls_meas   = '--'

for (season, months), col in zip(seasons.items(), colors_s):
    mask_s = df['month'].isin(months)
    g_pvgis = df.loc[mask_s].groupby('hour')['pvgis_kwh'].mean()
    g_meas  = df.loc[mask_s].groupby('hour')['measured_kwh'].mean()
    ax_diurnal.plot(g_pvgis.index, g_pvgis.values, color=col, lw=1.8, ls=ls_pvgis,
                    label=f'{season} — PVGIS')
    ax_diurnal.plot(g_meas.index,  g_meas.values,  color=col, lw=1.4, ls=ls_meas,
                    label=f'{season} — Measured')

ax_diurnal.set_xlabel('Hour of day', fontsize=10, fontweight='bold')
ax_diurnal.set_ylabel('Mean PV output (kWh/h)', fontsize=10, fontweight='bold')
ax_diurnal.set_title('Average Diurnal Profile by Season\n(solid = PVGIS, dashed = Measured)',
                     fontsize=11, fontweight='bold')
ax_diurnal.set_xticks(range(0, 24, 3))
ax_diurnal.legend(fontsize=7.5, frameon=True, ncol=2)

plt.savefig(
    r"C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\python\electri toy MILP\pvgis_vs_measured_2023.png",
    dpi=150, bbox_inches='tight'
)
plt.show()
print("\nPlot saved to pvgis_vs_measured_2023.png")
