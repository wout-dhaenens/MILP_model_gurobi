# -*- coding: utf-8 -*-
"""
Monthly heat demand comparison: BDEW model vs measured data
Months: Dec 2025, Jan 2026, Feb 2026 (and Mar 1-9)

Measured data  : Plot9.03.xlsx  (T_high, T_low, pump sensors)
BDEW profile   : shop_heat_demand_2025_2026.csv  (from demand_generation.py)
                 If the CSV does not exist yet, it is generated automatically.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

# ── SETTINGS ──────────────────────────────────────────────────────────────────
MEASURED_PATH  = r"C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\Plot9.03.xlsx"
BDEW_CSV       = "shop_heat_demand_2025_2026.csv"
GAS_CSV        = r"C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\gasdata.csv"
GAS_YEAR_B     = 2021   # season 2021/2022
BDEW_CSV_2021  = "shop_heat_demand_2021_2022.csv"   # BDEW with 2021/22 temperatures

FLOW_RATE_M3H  = 8.0
DENSITY_WATER  = 1000       # kg/m3
CP_WATER       = 4.186      # kJ/(kg·K)
mass_flow_rate = (FLOW_RATE_M3H * DENSITY_WATER) / 3600   # kg/s

MONTH_WINDOWS = [
    ("Dec 2025",    pd.Timestamp("2025-12-09"), pd.Timestamp("2025-12-31 23:00")),
    ("Jan 2026",    pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-31 23:00")),
    ("Feb 2026",    pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-28 23:00")),
    ("Mar 2026\n(1-9)", pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-09 23:00")),
]

# ── 1. LOAD BDEW PROFILE ──────────────────────────────────────────────────────
if not os.path.exists(BDEW_CSV):
    print(f"'{BDEW_CSV}' not found — generating now via demand_generation.py ...")
    from demand_generation import generate_season_profile, save_profile
    profile, temp_window = generate_season_profile(2025)
    save_profile(profile, temp_window, 2025)

df_bdew = pd.read_csv(BDEW_CSV, parse_dates=["datetime"])
df_bdew = df_bdew.set_index("datetime")
bdew_kw = df_bdew["heat_demand_kWh"]   # kW (hourly resolution, 1h → kWh = kW)
print(f"BDEW 2025/26 loaded: {bdew_kw.index[0]} to {bdew_kw.index[-1]}  ({len(bdew_kw)} h)")

# ── 1b. BDEW PROFILE WITH 2021/22 TEMPERATURES ───────────────────────────────
if not os.path.exists(BDEW_CSV_2021):
    print(f"'{BDEW_CSV_2021}' not found — generating via demand_generation.py ...")
    from demand_generation import generate_season_profile, save_profile
    profile_2021, temp_2021 = generate_season_profile(2021)
    save_profile(profile_2021, temp_2021, 2021)

df_bdew_2021 = pd.read_csv(BDEW_CSV_2021, parse_dates=["datetime"])
df_bdew_2021 = df_bdew_2021.set_index("datetime")
bdew_2021_kw = df_bdew_2021["heat_demand_kWh"]
print(f"BDEW 2021/22 loaded: {bdew_2021_kw.index[0]} to {bdew_2021_kw.index[-1]}  ({len(bdew_2021_kw)} h)")

# ── 2. LOAD MEASURED DATA (same as gasdata vs measured.py) ────────────────────
print(f"\nLoading measured data from {MEASURED_PATH} ...")
df_raw = pd.read_excel(MEASURED_PATH, header=0)

def load_sensor(df, ts_col, val_col):
    ts  = pd.to_datetime(df.iloc[:, ts_col], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    val = pd.to_numeric(df.iloc[:, val_col], errors="coerce")
    valid = ts.notna() & val.notna()
    return pd.Series(val.values[valid], index=ts.values[valid]).sort_index()

T_high = pd.concat([load_sensor(df_raw, 0, 1), load_sensor(df_raw, 2, 3)]).sort_index()
T_low  = pd.concat([load_sensor(df_raw, 4, 5), load_sensor(df_raw, 6, 7)]).sort_index()
pump   = pd.concat([load_sensor(df_raw, 8, 9), load_sensor(df_raw, 10, 11)]).sort_index()

df_sensors = pd.DataFrame({"T_high": T_high, "T_low": T_low, "pump": pump}).dropna()
df_sensors["Delta_T"]  = df_sensors["T_high"] - df_sensors["T_low"]
df_sensors["Power_kW"] = (mass_flow_rate * CP_WATER
                          * df_sensors["Delta_T"]
                          * df_sensors["pump"]).clip(lower=0)

measured_kw = df_sensors["Power_kW"].resample("h").mean()
print(f"Measured data loaded: {measured_kw.index.min():%d %b %Y} to {measured_kw.index.max():%d %b %Y}")

# ── 3. LOAD GAS DATA (same as gasdata vs measured.py) ────────────────────────
print(f"\nLoading gas data from {GAS_CSV} ...")
df_csv = pd.read_csv(
    GAS_CSV, sep=";", skiprows=1, header=None,
    names=["timestamp", "col2", "demand_kW_raw", "extra"],
    decimal=",", encoding="utf-8"
)
df_csv["timestamp"] = pd.to_datetime(df_csv["timestamp"], utc=True)
df_csv["timestamp"] = df_csv["timestamp"].dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
df_csv = df_csv.set_index("timestamp")

def gas_season_mwh(year, month_windows):
    """Extract monthly MWh totals for a Dec-Mar season from the gas CSV."""
    results = []
    for _, t_start, t_end in month_windows:
        # Map window month/day onto the gas data year
        y1, y2 = year, year + 1
        start_gas = t_start.replace(year=y1 if t_start.month == 12 else y2)
        end_gas   = t_end.replace(  year=y1 if t_end.month   == 12 else y2)
        sl = df_csv.loc[start_gas:end_gas, "demand_kW_raw"].dropna()
        sl_hourly = sl.resample("h").mean()
        results.append(sl_hourly.sum() / 1000)   # kWh -> MWh
    return results

gas_b_mwh = gas_season_mwh(GAS_YEAR_B, MONTH_WINDOWS)
print(f"  Gas {GAS_YEAR_B}/{GAS_YEAR_B+1}: {sum(gas_b_mwh):.1f} MWh total")

# ── 4. MONTHLY TOTALS (MWh) ───────────────────────────────────────────────────
# hourly kWh = kW * 1h  →  sum gives kWh  →  /1000 gives MWh
def monthly_mwh(series, t_start, t_end):
    """Sum hourly kW values in window → MWh. Strip tz if needed."""
    idx = series.index
    if hasattr(idx, "tz") and idx.tz is not None:
        t_start = t_start.tz_localize(idx.tz) if t_start.tzinfo is None else t_start
        t_end   = t_end.tz_localize(idx.tz)   if t_end.tzinfo   is None else t_end
    sl = series.loc[t_start:t_end].dropna()
    return sl.sum() / 1000   # kWh -> MWh

def remap_windows(windows, from_year, to_year):
    """Shift month-window timestamps from one season to another (e.g. 2025->2021)."""
    offset = to_year - from_year
    remapped = []
    for label, t_start, t_end in windows:
        remapped.append((
            label,
            t_start.replace(year=t_start.year + offset),
            t_end.replace(  year=t_end.year   + offset),
        ))
    return remapped

MONTH_WINDOWS_2021 = remap_windows(MONTH_WINDOWS, 2025, 2021)

months        = [w[0] for w in MONTH_WINDOWS]
bdew_mwh      = [monthly_mwh(bdew_kw,      w[1], w[2]) for w in MONTH_WINDOWS]
bdew_2021_mwh = [monthly_mwh(bdew_2021_kw, w[1], w[2]) for w in MONTH_WINDOWS_2021]
measured_mwh  = [monthly_mwh(measured_kw,  w[1], w[2]) for w in MONTH_WINDOWS]

print("\nMonthly totals [MWh]:")
print(f"  {'Month':<16} {'BDEW 25/26':>11} {'BDEW 21/22':>11} {'Measured':>10} "
      f"{'Gas '+str(GAS_YEAR_B)+'/'+str(GAS_YEAR_B+1):>14}")
print("  " + "-" * 66)
for m, b25, b21, meas, gb in zip(months, bdew_mwh, bdew_2021_mwh, measured_mwh, gas_b_mwh):
    print(f"  {m:<16} {b25:>11.1f} {b21:>11.1f} {meas:>10.1f} {gb:>14.1f}")

# ── 5. BAR PLOT ───────────────────────────────────────────────────────────────

series = [
    (bdew_mwh,      "BDEW model (temp 2025/26)",         "#d62728"),
    (bdew_2021_mwh, "BDEW model (temp 2021/22)",         "#9467bd"),
    (measured_mwh,  "Measured (2025/26)",                 "#1f77b4"),
    (gas_b_mwh,     f"Gas {GAS_YEAR_B}/{GAS_YEAR_B+1}",  "#ff7f0e"),
]

x       = np.arange(len(months))
width   = 0.18
offsets = [-1.5, -0.5, 0.5, 1.5]

fig, ax = plt.subplots(figsize=(15, 6))

for (values, label, color), offset in zip(series, offsets):
    bars = ax.bar(x + offset * width, values, width,
                  label=label, color=color, alpha=0.85,
                  edgecolor="white", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7.5,
                    color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(months, fontsize=11)
ax.set_ylabel("Thermal energy [MWh]", fontsize=12)
ax.set_title("Monthly heat demand comparison\n"
             "BDEW (temp 2025/26 & 2021/22)  |  Measured 2025/26  |  Gas profile 2021/22",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, max(max(v) for v, *_ in series) * 1.22)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.grid(axis="x", visible=False)

plt.tight_layout()
out_png = "heat_demand_comparison_2025_2026.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nBar chart saved --> {out_png}")
plt.show()

# ── 6. TEMPERATURE COMPARISON PLOT ───────────────────────────────────────────
temp_2526 = df_bdew["temperature_C"]
temp_2122 = df_bdew_2021["temperature_C"]

# Build a common x-axis as "days since Dec 9" so both seasons overlap
def days_since_dec9(index):
    origin = index[0].replace(hour=0, minute=0, second=0)
    return (index - origin).total_seconds() / 3600  # hours offset

x_2526 = days_since_dec9(temp_2526.index)
x_2122 = days_since_dec9(temp_2122.index)

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Panel 1: overlay both temperature series
ax1.plot(x_2526, temp_2526.values, color="#d62728", linewidth=0.8,
         alpha=0.9, label="2025/26 (open-meteo)")
ax1.plot(x_2122, temp_2122.values, color="#1f77b4", linewidth=0.8,
         alpha=0.9, label="2021/22 (PVGIS)")
ax1.axhline(0, color="k", linewidth=0.5, linestyle="--")
ax1.set_ylabel("Outdoor temperature [°C]", fontsize=11)
ax1.set_title("Outdoor temperature comparison: season 2025/26 vs 2021/22\n"
              "(hours since Dec 9)", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# Panel 2: difference (2025/26 minus 2021/22)
# Align on common index (same length since both are 2184 h)
min_len = min(len(temp_2526), len(temp_2122))
diff_temp = temp_2526.values[:min_len] - temp_2122.values[:min_len]
ax2.bar(x_2526[:min_len], diff_temp, width=1.0,
        color=["#d62728" if v > 0 else "#1f77b4" for v in diff_temp],
        alpha=0.6)
ax2.axhline(0, color="k", linewidth=0.8)
ax2.set_ylabel("Temperature difference [°C]\n(2025/26 minus 2021/22)", fontsize=10)
ax2.set_xlabel("Hours since Dec 9", fontsize=11)
ax2.grid(axis="y", linestyle="--", alpha=0.4)

from matplotlib.patches import Patch
ax2.legend(handles=[
    Patch(color="#d62728", alpha=0.6, label="2025/26 warmer"),
    Patch(color="#1f77b4", alpha=0.6, label="2021/22 warmer"),
], fontsize=9)

# Monthly mean temperature summary
print("\nMonthly mean temperatures [°C]:")
print(f"  {'Month':<16} {'2025/26':>10} {'2021/22':>10} {'Diff':>8}")
print("  " + "-" * 48)
for label, t_start, t_end in MONTH_WINDOWS:
    t2526 = temp_2526.loc[t_start:t_end].mean()
    remap = MONTH_WINDOWS_2021[MONTH_WINDOWS.index((label, t_start, t_end))]
    t2122 = temp_2122.loc[remap[1]:remap[2]].mean()
    print(f"  {label:<16} {t2526:>10.1f} {t2122:>10.1f} {t2526-t2122:>+8.1f}")

plt.tight_layout()
out_png2 = "temperature_comparison_2025_2026_vs_2021_2022.png"
fig2.savefig(out_png2, dpi=150, bbox_inches="tight")
print(f"Temperature plot saved --> {out_png2}")
plt.show()
