"""
compare_azimuth.py
──────────────────
Fetches PVGIS for several azimuth values and overlays their average
summer diurnal profile against measured data to find the best azimuth.
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIG
# ==============================================================================
LAT, LON, YEAR, TILT = 51.18, 3.55, 2023, 35
INSTALLED_KWP = 34.63
AZIMUTHS = [0, 20, 30, 40, 50]   # west-of-south degrees to test

MEASURED_CSV = (
    r"C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\python\electri toy MILP"
    r"\wetransfer_historiek_submeting_elektriciteit_541454897100022743_20250101_20260101"
    r"_kwartiertotalen-csv_2026-03-26_1550"
    r"\Historiek_submeting_elektriciteit_541454897100022743_20230101_20240101_kwartiertotalen.csv"
)
SUMMER_MONTHS = [6, 7, 8]   # June, July, August — clearest signal

# ==============================================================================
# LOAD MEASURED
# ==============================================================================
df_raw = pd.read_csv(MEASURED_CSV, sep=';', decimal=',', encoding='utf-8-sig')
df_pv  = df_raw[df_raw['Register'] == 'Productie Actief'].copy()
df_pv['timestamp'] = pd.to_datetime(
    df_pv['Van (datum)'] + ' ' + df_pv['Van (tijdstip)'], dayfirst=True)
df_pv = df_pv.set_index('timestamp').sort_index()
df_pv['Volume'] = pd.to_numeric(df_pv['Volume'], errors='coerce')
measured_hourly = df_pv['Volume'].resample('h').sum()

summer_mask = measured_hourly.index.month.isin(SUMMER_MONTHS)
measured_summer = measured_hourly[summer_mask]
diurnal_measured = measured_summer.groupby(measured_summer.index.hour).mean()

# ==============================================================================
# FETCH PVGIS FOR EACH AZIMUTH
# ==============================================================================
def fetch_pvgis(azimuth):
    url = (f"https://re.jrc.ec.europa.eu/api/seriescalc?"
           f"lat={LAT}&lon={LON}&startyear={YEAR}&endyear={YEAR}"
           f"&pvcalculation=1&peakpower=1.0"
           f"&angle={TILT}&aspect={azimuth}"
           f"&loss=14&outputformat=json")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data['outputs']['hourly'])
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
    df['time'] = df['time'].dt.floor('h')
    df = df.set_index('time')
    return df['P'] / 1000.0 * INSTALLED_KWP   # kWh/h

# ==============================================================================
# PLOT
# ==============================================================================
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

fig, ax = plt.subplots(figsize=(11, 6))

# Measured
ax.plot(diurnal_measured.index, diurnal_measured.values,
        color='black', lw=2.5, ls='--', label='Measured', zorder=5)

for az, col in zip(AZIMUTHS, colors):
    print(f"Fetching azimuth={az}...")
    pv = fetch_pvgis(az)
    summer_pv = pv[pv.index.month.isin(SUMMER_MONTHS)]
    diurnal_pv = summer_pv.groupby(summer_pv.index.hour).mean()
    ax.plot(diurnal_pv.index, diurnal_pv.values,
            color=col, lw=1.8, label=f'PVGIS azimuth={az}°')

ax.set_xlabel('Hour of day', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean PV output (kWh/h)  —  Summer (Jun–Aug)', fontsize=12, fontweight='bold')
ax.set_title('Azimuth sensitivity: average summer diurnal profile vs measured\n'
             '(best azimuth = solid line that best follows the dashed measured line)',
             fontsize=13, fontweight='bold')
ax.set_xticks(range(0, 24, 1))
ax.set_xlim(5, 21)
ax.legend(fontsize=10, frameon=True)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    r"C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\python\electri toy MILP\azimuth_comparison.png",
    dpi=150, bbox_inches='tight'
)
plt.show()
print("Done.")
