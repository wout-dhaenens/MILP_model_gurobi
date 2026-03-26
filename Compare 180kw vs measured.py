import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-v0_8-whitegrid')

# ── Constants ─────────────────────────────────────────────────────────────────
FLOW_RATE_M3H  = 8.0
DENSITY_WATER  = 1000        # kg/m³
CP_WATER       = 4.186       # kJ/(kg·K)
mass_flow_rate = (FLOW_RATE_M3H * DENSITY_WATER) / 3600   # kg/s

MEASURED_PATH = r"C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\Plot20.12.xlsx"
DEMAND_PATH   = r"C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\thermal_demand_profile_180kWth.xlsx"

# ── 1. Load & process measured data ───────────────────────────────────────────
df_raw = pd.read_excel(MEASURED_PATH)
df_raw.columns = df_raw.columns.str.strip()
df_raw['TimeString'] = pd.to_datetime(df_raw['TimeString'], format='%d.%m.%Y %H:%M:%S')
df_raw = df_raw.set_index('TimeString')

df_raw['Delta_T']  = df_raw['VarValue_high'] - df_raw['VarValue_low']
df_raw['Power_kW'] = mass_flow_rate * CP_WATER * df_raw['Delta_T'] * df_raw['VarValue_pump']
df_raw['Power_kW'] = df_raw['Power_kW'].clip(lower=0)   # remove negative artefacts

# Resample to hourly mean
measured_hourly = df_raw['Power_kW'].resample('h').mean()

# ── 2. Load theoretical demand profile (full year, hourly) ────────────────────
df_demand = pd.read_excel(DEMAND_PATH, sheet_name='Hourly Profile')
df_demand['Timestamp'] = pd.to_datetime(df_demand['Timestamp'])
df_demand = df_demand.set_index('Timestamp')
demand_hourly = df_demand['Thermal Demand (kW_th)']

# ── 3. Align on the overlapping window (Dec 4–20) ─────────────────────────────
start = measured_hourly.index.min().floor('h')
end   = measured_hourly.index.max().ceil('h')

demand_window   = demand_hourly.loc[start:end]
measured_window = measured_hourly.reindex(demand_window.index)   # align index

# ── 4. Statistics ─────────────────────────────────────────────────────────────
diff = measured_window - demand_window

print("=" * 58)
print("   COMPARISON: Measured Power  vs  Theoretical Demand")
print("=" * 58)
print(f"  Period          : {start:%d %b %Y %H:%M}  →  {end:%d %b %Y %H:%M}")
print(f"  Hours compared  : {demand_window.notna().sum()}")
print()
print(f"  {'Metric':<28} {'Theoretical':>12} {'Measured':>12}")
print("  " + "-" * 54)
print(f"  {'Peak power (kW)':<28} {demand_window.max():>12.1f} {measured_window.max():>12.1f}")
print(f"  {'Mean power (kW)':<28} {demand_window.mean():>12.1f} {measured_window.mean():>12.1f}")
print(f"  {'Min power (kW)':<28} {demand_window.min():>12.1f} {measured_window.min():>12.1f}")
print(f"  {'Total energy (kWh)':<28} {demand_window.sum():>12.0f} {measured_window.sum():>12.0f}")
print()
print(f"  Mean difference (measured − theoretical): {diff.mean():+.1f} kW")
print(f"  RMSE                                    : {np.sqrt((diff**2).mean()):.1f} kW")
print(f"  Coverage (measured / theoretical)       : {100*measured_window.sum()/demand_window.sum():.1f} %")
print("=" * 58)

# ── 5. Plotting ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 13))
gs  = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32,
               height_ratios=[2.2, 1.4, 1.4])

ax_main  = fig.add_subplot(gs[0, :])    # full-width time series
ax_diff  = fig.add_subplot(gs[1, :])    # difference over time
ax_box   = fig.add_subplot(gs[2, 0])    # box-plot comparison
ax_scatter = fig.add_subplot(gs[2, 1])  # scatter: theoretical vs measured

date_fmt = mdates.DateFormatter("%d/%m")

# ── Panel 1 : Overlay time series ────────────────────────────────────────────
ax_main.plot(demand_window.index,   demand_window,   color='#d62728', lw=1.8,
             label='Theoretical demand profile', zorder=3)
ax_main.plot(measured_window.index, measured_window, color='#1f77b4', lw=1.5,
             alpha=0.85, label='Measured power (from ΔT & pump)', zorder=2)
ax_main.fill_between(demand_window.index, demand_window, measured_window,
                     where=(measured_window < demand_window),
                     interpolate=True, alpha=0.15, color='#d62728',
                     label='Unmet demand')
ax_main.fill_between(demand_window.index, demand_window, measured_window,
                     where=(measured_window >= demand_window),
                     interpolate=True, alpha=0.15, color='#1f77b4',
                     label='Excess delivery')
ax_main.set_ylabel('Thermal Power (kW)', fontsize=11, fontweight='bold')
ax_main.set_title('Measured Thermal Power  vs  Theoretical Demand Profile\n'
                  '(Dec 4 – Dec 20, 2025)', fontsize=14, fontweight='bold', pad=10)
ax_main.legend(loc='upper left', fontsize=9, frameon=True)
ax_main.xaxis.set_major_formatter(date_fmt)
ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=30)

# ── Panel 2 : Difference (measured − theoretical) ────────────────────────────
ax_diff.axhline(0, color='black', lw=1, ls='--')
ax_diff.bar(diff.index, diff.values, width=1/24, align='center',
            color=np.where(diff.values >= 0, '#1f77b4', '#d62728'), alpha=0.7)
ax_diff.set_ylabel('Difference (kW)\nmeasured − theoretical', fontsize=10, fontweight='bold')
ax_diff.set_title('Hourly Difference: Measured − Theoretical', fontsize=11, fontweight='bold')
ax_diff.xaxis.set_major_formatter(date_fmt)
ax_diff.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax_diff.xaxis.get_majorticklabels(), rotation=30)

# Add legend patches for colours
from matplotlib.patches import Patch
ax_diff.legend(handles=[
    Patch(color='#1f77b4', alpha=0.7, label='Excess delivery'),
    Patch(color='#d62728', alpha=0.7, label='Unmet demand'),
], fontsize=9, frameon=True)

# ── Panel 3 : Box-plot ────────────────────────────────────────────────────────
bp = ax_box.boxplot(
    [demand_window.dropna().values, measured_window.dropna().values],
    labels=['Theoretical\ndemand', 'Measured\npower'],
    patch_artist=True,
    medianprops=dict(color='black', lw=2),
    whiskerprops=dict(lw=1.5),
    capprops=dict(lw=1.5),
)
bp['boxes'][0].set_facecolor('#d62728'); bp['boxes'][0].set_alpha(0.5)
bp['boxes'][1].set_facecolor('#1f77b4'); bp['boxes'][1].set_alpha(0.5)
ax_box.set_ylabel('Thermal Power (kW)', fontsize=10, fontweight='bold')
ax_box.set_title('Distribution Comparison', fontsize=11, fontweight='bold')

# ── Panel 4 : Scatter ─────────────────────────────────────────────────────────
mask = demand_window.notna() & measured_window.notna()
x = demand_window[mask].values
y = measured_window[mask].values
ax_scatter.scatter(x, y, alpha=0.35, s=15, color='#9467bd', edgecolors='none')
lim = max(x.max(), y.max()) * 1.05
ax_scatter.plot([0, lim], [0, lim], 'k--', lw=1.5, label='1:1 line')
# Linear fit
m, b = np.polyfit(x, y, 1)
xfit = np.linspace(0, lim, 200)
ax_scatter.plot(xfit, m*xfit + b, color='#ff7f0e', lw=1.8,
                label=f'Fit: y = {m:.2f}x + {b:.1f}')
ax_scatter.set_xlabel('Theoretical demand (kW)', fontsize=10, fontweight='bold')
ax_scatter.set_ylabel('Measured power (kW)',     fontsize=10, fontweight='bold')
ax_scatter.set_title('Scatter: Theoretical vs Measured', fontsize=11, fontweight='bold')
ax_scatter.set_xlim(0, lim); ax_scatter.set_ylim(0, lim)
ax_scatter.legend(fontsize=9, frameon=True)
r2 = np.corrcoef(x, y)[0, 1]**2
ax_scatter.text(0.05, 0.92, f'R² = {r2:.3f}', transform=ax_scatter.transAxes,
                fontsize=10, fontweight='bold')

plt.savefig('/mnt/user-data/outputs/demand_vs_measured_comparison.png',
            dpi=300, bbox_inches='tight')
plt.show()
print("\nPlot saved to demand_vs_measured_comparison.png")