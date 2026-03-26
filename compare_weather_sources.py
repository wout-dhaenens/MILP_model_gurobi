# -*- coding: utf-8 -*-
"""
Compare PVGIS vs open-meteo temperature data for the same year.
Uses the fetch functions from demand_generation.py directly.
Run this to check whether the two sources agree — if they don't,
it explains why BDEW gives different totals for different seasons.
"""

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from demand_generation import (
    fetch_temperature_pvgis,
    fetch_temperature_openmeteo,
    LATITUDE, LONGITUDE,
)

plt.style.use('seaborn-v0_8-whitegrid')

COMPARE_YEAR = 2021   # must be available in PVGIS (~2 yr lag)
SEASON_START = pd.Timestamp(f"{COMPARE_YEAR}-12-09")
SEASON_END   = pd.Timestamp(f"{COMPARE_YEAR + 1}-03-09 23:00")

# ── 1. Fetch both sources ─────────────────────────────────────────────────────
print(f"Fetching PVGIS temperature {COMPARE_YEAR} ...")
pvgis_y1     = fetch_temperature_pvgis(COMPARE_YEAR)
print(f"Fetching PVGIS temperature {COMPARE_YEAR + 1} ...")
pvgis_y2     = fetch_temperature_pvgis(COMPARE_YEAR + 1)

print(f"Fetching open-meteo temperature {COMPARE_YEAR} ...")
openmeteo_y1 = fetch_temperature_openmeteo(COMPARE_YEAR)
print(f"Fetching open-meteo temperature {COMPARE_YEAR + 1} ...")
openmeteo_y2 = fetch_temperature_openmeteo(COMPARE_YEAR + 1)

# ── 2. Slice to Dec 9 – Mar 9 window ─────────────────────────────────────────
cut1 = pd.Timestamp(f"{COMPARE_YEAR}-12-31 23:00")
cut2 = pd.Timestamp(f"{COMPARE_YEAR + 1}-01-01 00:00")

pvgis_season     = pd.concat([pvgis_y1[SEASON_START:cut1],     pvgis_y2[cut2:SEASON_END]])
openmeteo_season = pd.concat([openmeteo_y1[SEASON_START:cut1], openmeteo_y2[cut2:SEASON_END]])

# Align on common index
common_idx   = pvgis_season.index.intersection(openmeteo_season.index)
pvgis_al     = pvgis_season.reindex(common_idx)
openmeteo_al = openmeteo_season.reindex(common_idx)
diff         = openmeteo_al - pvgis_al

print(f"\nTemperature comparison over Dec {COMPARE_YEAR} – Mar {COMPARE_YEAR+1}:")
print(f"  PVGIS      mean: {pvgis_al.mean():.2f} °C  |  std: {pvgis_al.std():.2f} °C")
print(f"  open-meteo mean: {openmeteo_al.mean():.2f} °C  |  std: {openmeteo_al.std():.2f} °C")
print(f"  Difference mean: {diff.mean():+.2f} °C  |  RMSE: {np.sqrt((diff**2).mean()):.2f} °C")
print(f"  Max positive diff (open-meteo warmer): +{diff.max():.1f} °C")
print(f"  Max negative diff (PVGIS warmer)     : {diff.min():.1f} °C")

# ── 3. Plots ──────────────────────────────────────────────────────────────────
hours = np.arange(len(common_idx))

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle(
    f"PVGIS vs open-meteo temperature — Dec {COMPARE_YEAR} to Mar {COMPARE_YEAR+1}\n"
    f"Location: lat={LATITUDE}, lon={LONGITUDE}",
    fontsize=13, fontweight="bold"
)

# Panel 1: overlay
axes[0].plot(hours, pvgis_al.values,     color="#1f77b4", lw=0.8, alpha=0.9, label="PVGIS")
axes[0].plot(hours, openmeteo_al.values, color="#d62728", lw=0.8, alpha=0.9, label="open-meteo")
axes[0].axhline(0, color="k", lw=0.5, ls="--")
axes[0].set_ylabel("Temperature [°C]")
axes[0].set_title("Overlay", fontsize=11)
axes[0].legend(fontsize=10)

# Panel 2: difference
axes[1].bar(hours, diff.values, width=1.0,
            color=["#d62728" if v > 0 else "#1f77b4" for v in diff.values],
            alpha=0.6)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_ylabel("Difference [°C]\n(open-meteo minus PVGIS)")
axes[1].set_title(f"Hourly difference  (mean={diff.mean():+.2f} °C, RMSE={np.sqrt((diff**2).mean()):.2f} °C)",
                  fontsize=11)
from matplotlib.patches import Patch
axes[1].legend(handles=[
    Patch(color="#d62728", alpha=0.6, label="open-meteo warmer"),
    Patch(color="#1f77b4", alpha=0.6, label="PVGIS warmer"),
], fontsize=9)

# Panel 3: scatter
axes[2].remove()
ax_sc = fig.add_subplot(3, 1, 3)
ax_sc.scatter(pvgis_al.values, openmeteo_al.values,
              s=2, alpha=0.3, color="#7f7f7f")
lim = max(abs(pvgis_al.min()), abs(pvgis_al.max()),
          abs(openmeteo_al.min()), abs(openmeteo_al.max())) * 1.05
ax_sc.plot([-lim, lim], [-lim, lim], "k--", lw=1.2, label="1:1 line")
m, b = np.polyfit(pvgis_al.values, openmeteo_al.values, 1)
xfit = np.linspace(-lim, lim, 200)
ax_sc.plot(xfit, m * xfit + b, color="#d62728", lw=1.5,
           label=f"Fit: y = {m:.3f}x + {b:.2f}")
r2 = np.corrcoef(pvgis_al.values, openmeteo_al.values)[0, 1] ** 2
ax_sc.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax_sc.transAxes,
           fontsize=10, fontweight="bold")
ax_sc.set_xlabel("PVGIS temperature [°C]")
ax_sc.set_ylabel("open-meteo temperature [°C]")
ax_sc.set_title("Scatter: PVGIS vs open-meteo", fontsize=11)
ax_sc.legend(fontsize=9)
ax_sc.set_xlim(-lim, lim)
ax_sc.set_ylim(-lim, lim)

plt.tight_layout()
out_png = f"weather_source_comparison_{COMPARE_YEAR}.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nPlot saved --> {out_png}")
plt.show()
