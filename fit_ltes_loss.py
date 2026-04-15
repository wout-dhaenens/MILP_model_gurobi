# -*- coding: utf-8 -*-
"""
fit_ltes_loss.py
================
Fits standby losses from Sunamp Thermino e datasheet.

Linear model:  loss_hr = a + b * cap
  a [kWh/h]          fixed loss when storage is installed
  b [kWh/h/kWh_th]   proportional loss rate (LTES_LOSS_FRAC_HR in MILP)

Data source: Sunamp Thermino e technical datasheet
  Model   | Capacity (kWh) | Heat loss (kWh/24 h)
  --------|----------------|---------------------
  70 e    |   3.5          |   0.48
  150 e   |   7.0          |   0.68
  210 e   |  10.5          |   0.77
  300 e   |  14.0          |   0.84
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ==============================================================================
# --- DATA ---
# ==============================================================================

models      = ["70 e", "150 e", "210 e", "300 e"]
capacity    = np.array([3.5,  7.0,  10.5, 14.0])   # kWh_th
loss_per_24 = np.array([0.48, 0.68,  0.77,  0.84])  # kWh / 24 h
loss_per_hr = loss_per_24 / 24.0                     # kWh/h

# ==============================================================================
# --- LINEAR FIT:  loss_hr = a + b * cap ---
# ==============================================================================

b_lin, a_lin, r_lin, _, _ = linregress(capacity, loss_per_hr)
r2_lin   = r_lin**2
pred_lin = a_lin + b_lin * capacity
res_lin  = loss_per_hr - pred_lin
rmse_lin = np.sqrt(np.mean(res_lin**2))

print("=" * 60)
print(f"  Linear fit:  loss_hr = a + b * cap")
print(f"    a (fixed)  = {a_lin:.6f} kWh/h  ({a_lin*1000:.3f} Wh/h)")
print(f"    b (frac)   = {b_lin:.6f} /h     ({b_lin*1000:.4f} Wh/h per kWh_th)")
print(f"    R2         = {r2_lin:.4f}")
print(f"    RMSE       = {rmse_lin*1000:.4f} Wh/h")
print("=" * 60)
print(f"\n  MILP parameters:")
print(f"    LTES_LOSS_FIXED_HR = {a_lin:.6f}  kWh/h")
print(f"    LTES_LOSS_FRAC_HR  = {b_lin:.6f}  /h")

# ==============================================================================
# --- PLOT ---
# ==============================================================================

cap_fine = np.linspace(0, 17, 300)
loss_fit = a_lin + b_lin * cap_fine

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(capacity, loss_per_hr * 1000, s=90, color='black', zorder=6,
           label='Thermino e data')
ax.plot(cap_fine, loss_fit * 1000, color='steelblue', lw=2,
        label=(f'Linear fit  (R$^2$={r2_lin:.4f})\n'
               f'$a$ = {a_lin*1000:.2f} Wh/h,  '
               f'$b$ = {b_lin*1000:.3f} Wh/h per kWh$_{{th}}$'))

for xi, yi, mi in zip(capacity, loss_per_hr * 1000, models):
    ax.annotate(mi, (xi, yi), textcoords='offset points', xytext=(6, 4), fontsize=9)

ax.set_xlim(0, 17)
ax.set_ylim(0, None)
ax.set_xlabel("LTES capacity  (kWh$_{th}$)", fontsize=11)
ax.set_ylabel("Hourly standby loss  (Wh / h)", fontsize=11)
ax.set_title("LTES standby loss vs capacity  (Sunamp Thermino e)", fontsize=11)
ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.02, 0.98))
ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig("ltes_loss_fit.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to ltes_loss_fit.png")
