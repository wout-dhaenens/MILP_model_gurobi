import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-v0_8-whitegrid')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these paths and year choices
# ══════════════════════════════════════════════════════════════════════════════
MEASURED_PATH = r"C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\Plot9.03.xlsx"
CSV_PATH      = r"C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\gasdata.csv"

YEAR_A = 2018   # ← first reference year  (2008–2022 available)
YEAR_B = 2021   # ← second reference year (2008–2022 available)
# ══════════════════════════════════════════════════════════════════════════════

# ── Constants ─────────────────────────────────────────────────────────────────
FLOW_RATE_M3H  = 8.0
DENSITY_WATER  = 1000        # kg/m³
CP_WATER       = 4.186       # kJ/(kg·K)
mass_flow_rate = (FLOW_RATE_M3H * DENSITY_WATER) / 3600   # kg/s

# ── 1. Load & process measured data ───────────────────────────────────────────
df_raw = pd.read_excel(MEASURED_PATH, header=0)

def load_sensor(df, ts_col, val_col):
    ts  = pd.to_datetime(df.iloc[:, ts_col], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    val = pd.to_numeric(df.iloc[:, val_col], errors='coerce')
    valid = ts.notna() & val.notna()
    return pd.Series(val.values[valid], index=ts.values[valid]).sort_index()

T_high = pd.concat([load_sensor(df_raw, 0, 1), load_sensor(df_raw, 2, 3)]).sort_index()
T_low  = pd.concat([load_sensor(df_raw, 4, 5), load_sensor(df_raw, 6, 7)]).sort_index()
pump   = pd.concat([load_sensor(df_raw, 8, 9), load_sensor(df_raw, 10, 11)]).sort_index()

df_sensors = pd.DataFrame({'T_high': T_high, 'T_low': T_low, 'pump': pump})
df_sensors = df_sensors.dropna()

df_sensors['Delta_T']  = df_sensors['T_high'] - df_sensors['T_low']
df_sensors['Power_kW'] = (mass_flow_rate * CP_WATER
                          * df_sensors['Delta_T']
                          * df_sensors['pump'])
df_sensors['Power_kW'] = df_sensors['Power_kW'].clip(lower=0)
measured_hourly = df_sensors['Power_kW'].resample('h').mean()

print(f"Measured data loaded: {len(df_sensors)} records, "
      f"{measured_hourly.index.min():%d %b %Y} – {measured_hourly.index.max():%d %b %Y}")

meas_start = pd.Timestamp('2025-12-09 00:00:00')
meas_end   = pd.Timestamp('2026-03-09 23:00:00')
measured_window = measured_hourly.loc[meas_start:meas_end]

meas_month = measured_window.index.month
meas_day   = measured_window.index.day
meas_hour  = measured_window.index.hour

# ── 2. Load CSV gas demand profile ────────────────────────────────────────────
df_csv = pd.read_csv(
    CSV_PATH,
    sep=';',
    skiprows=1,
    header=None,
    names=['timestamp', 'col2', 'demand_kW_raw', 'extra'],
    decimal=',',
    encoding='utf-8'
)
df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'], utc=True)
df_csv['timestamp'] = df_csv['timestamp'].dt.tz_convert('Europe/Brussels').dt.tz_localize(None)
df_csv = df_csv.set_index('timestamp')
df_csv['demand_kW'] = df_csv['demand_kW_raw']

def extract_window(year):
    mask = (
        (
            (df_csv.index.year == year) &
            (df_csv.index.month == 12) &
            (df_csv.index.day >= 9)
        ) | (
            (df_csv.index.year == year + 1) &
            (df_csv.index.month.isin([1, 2])) |
            (
                (df_csv.index.year == year + 1) &
                (df_csv.index.month == 3) &
                (df_csv.index.day <= 9)
            )
        )
    )
    sub = df_csv.loc[mask, 'demand_kW'].resample('h').mean()
    lookup = {(r.month, r.day, r.hour): v for r, v in sub.items()}
    values = [
        lookup.get((m, d, h), np.nan)
        for m, d, h in zip(meas_month, meas_day, meas_hour)
    ]
    return pd.Series(values, index=measured_window.index, name=f'CSV {year}/{year+1}')

csv_a = extract_window(YEAR_A)
csv_b = extract_window(YEAR_B)

# ── 3. Define 3 monthly windows ───────────────────────────────────────────────
# Month 1: Dec 9 – Dec 31 2025
# Month 2: Jan 1 – Jan 31 2026
# Month 3: Feb 1 – Mar 9 2026
month_windows = [
    ('Dec 2025',  pd.Timestamp('2025-12-09'), pd.Timestamp('2025-12-31 23:00')),
    ('Jan 2026',  pd.Timestamp('2026-01-01'), pd.Timestamp('2026-01-31 23:00')),
    ('Feb–Mar 2026', pd.Timestamp('2026-02-01'), pd.Timestamp('2026-03-09 23:00')),
]

COLORS = {
    'measured': '#1f77b4',
    'year_a':   '#d62728',
    'year_b':   '#2ca02c',
}

# ── 4. Plot — 3 monthly subplots ──────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=False)
fig.suptitle(
    f'Measured Power (Dec 2025 – Mar 2026)  vs  CSV Demand — {YEAR_A}/{YEAR_A+1} & {YEAR_B}/{YEAR_B+1}\n'
    f'(day/hour matched)',
    fontsize=13, fontweight='bold', y=1.01
)

date_fmt = mdates.DateFormatter("%d/%m")

for ax, (month_label, t_start, t_end) in zip(axes, month_windows):
    # Slice each series to this month's window
    m_slice   = measured_window.loc[t_start:t_end]
    csv_a_sl  = csv_a.loc[t_start:t_end]
    csv_b_sl  = csv_b.loc[t_start:t_end]

    ax.plot(csv_a_sl.index, csv_a_sl.values,
            color=COLORS['year_a'], lw=1.8, label=f'CSV {YEAR_A}/{YEAR_A+1}', zorder=3)
    ax.plot(csv_b_sl.index, csv_b_sl.values,
            color=COLORS['year_b'], lw=1.8, ls='--', label=f'CSV {YEAR_B}/{YEAR_B+1}', zorder=3)
    ax.plot(m_slice.index, m_slice.values,
            color=COLORS['measured'], lw=1.5, alpha=0.9,
            label='Measured power (ΔT & pump)', zorder=4)

    ax.set_title(month_label, fontsize=11, fontweight='bold', pad=6)
    ax.set_ylabel('Thermal Power (kW)', fontsize=10)
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    # ── Stats text box ────────────────────────────────────────────────────────
    def stats_line(label, s, color):
        s = s.dropna()
        return (f"{label}:  mean {s.mean():.1f} kW  |  "
                f"peak {s.max():.1f} kW  |  "
                f"total {s.sum():.0f} kWh")

    stats_text = "\n".join([
        stats_line(f"Measured      ", m_slice,  COLORS['measured']),
        stats_line(f"CSV {YEAR_A}/{YEAR_A+1}", csv_a_sl, COLORS['year_a']),
        stats_line(f"CSV {YEAR_B}/{YEAR_B+1}", csv_b_sl, COLORS['year_b']),
    ])
    ax.text(0.01, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#cccccc', alpha=0.85))

    # Also print to console
    print(f"\n── {month_label} ──")
    for label, s in [(f"Measured      ", m_slice),
                     (f"CSV {YEAR_A}/{YEAR_A+1}", csv_a_sl),
                     (f"CSV {YEAR_B}/{YEAR_B+1}", csv_b_sl)]:
        s = s.dropna()
        print(f"  {label}  mean={s.mean():.1f} kW  peak={s.max():.1f} kW  total={s.sum():.0f} kWh")

    ax.legend(loc='upper right', fontsize=9, frameon=True)

    # Set x-limits tightly to the month
    ax.set_xlim(t_start, t_end)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/demand_vs_measured_3months.png',
            dpi=300, bbox_inches='tight')
plt.show()
print("\nPlot saved to demand_vs_measured_3months.png")