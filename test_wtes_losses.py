# -*- coding: utf-8 -*-

"""
test_wtes_losses.py
--------------------
Standalone script to inspect and visualise the WTES (stratified water-tank TES)
loss model used in Milp_yearly_test_gurobi.py.

Loss model per hour:
    loss_t = beta_wtes  * Q_wtes[t-1]   (proportional wall loss)
           + gamma_wtes * C_WTES        (ambient wall loss at reference ΔT)
           + loss_lids_wtes             (constant lid/top-bottom loss)

Physical parameters are identical to the main MILP file so results are
directly comparable.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Physical parameters (must match Milp_yearly_test_gurobi.py)
# ==============================================================================
temp_h   = 60.0   # °C supply
temp_c   = 40.0   # °C return
temp_env = 10.0   # °C ambient

T_HIGH = temp_h
T_LOW  = temp_c

delta_T_HC = T_HIGH - T_LOW        # 20 K
delta_T_C0 = T_LOW  - temp_env     # 30 K
delta_T_H0 = T_HIGH - temp_env     # 50 K

rho_water = 971.8    # kg/m³
c_water   = 4190.0   # J/(kg·K)
dt_sec    = 3600     # s/h

d_wtes       = 1.0                                      # m  (diameter)
A_cross_wtes = math.pi * (d_wtes / 2) ** 2             # m²

U_wall_wtes = 0.4   # W/(m²·K)

kWh_per_m_wtes = (A_cross_wtes * rho_water * c_water * delta_T_HC) / 3.6e6  # kWh/m

beta_wtes  = U_wall_wtes * (4.0 / (d_wtes * rho_water * c_water)) * dt_sec 
gamma_wtes = beta_wtes * (delta_T_C0 / delta_T_HC)
loss_lids_wtes = (
    U_wall_wtes * 2.0 * A_cross_wtes
    * ((delta_T_H0 + delta_T_C0) / 2.0)
    * dt_sec / 3.6e6
)

print("=== WTES loss parameters ===")
print(f"  d_wtes          = {d_wtes:.2f} m")
print(f"  A_cross_wtes    = {A_cross_wtes:.4f} m²")
print(f"  kWh_per_m_wtes  = {kWh_per_m_wtes:.4f} kWh/m")
print(f"  beta_wtes       = {beta_wtes:.6f}  (proportional wall-loss fraction per hour)")
print(f"  gamma_wtes      = {gamma_wtes:.6f}  (capacity-proportional ambient-loss fraction per hour)")
print(f"  loss_lids_wtes  = {loss_lids_wtes:.6f} kWh/h  (constant lid loss)")
print()

# ==============================================================================
# 1. Loss rate as a function of tank height (= capacity proxy)
# ==============================================================================
h_range   = np.linspace(0.1, 25.0, 200)          # m
C_range   = kWh_per_m_wtes * h_range              # kWh_th

# Evaluate at SoC = 50 % (mid-charge)
soc_frac  = 0.5
Q_at_50   = soc_frac * C_range

loss_prop = beta_wtes * Q_at_50                   # proportional term [kWh/h]
loss_amb  = gamma_wtes * C_range                  # ambient / capacity term [kWh/h]
loss_lid  = np.full_like(C_range, loss_lids_wtes) # constant term [kWh/h]
loss_total = loss_prop + loss_amb + loss_lid

loss_frac  = loss_total / C_range * 100           # % of capacity lost per hour

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("WTES loss model — sensitivity to tank size", fontsize=13)

ax = axes[0]
ax.plot(h_range, loss_prop,  label="Proportional wall loss  (β·Q)")
ax.plot(h_range, loss_amb,   label="Ambient wall loss         (γ·C)")
ax.plot(h_range, loss_lid,   label="Lid loss                         (const)")
ax.plot(h_range, loss_total, "k--", linewidth=1.8, label="Total loss")
ax.set_xlabel("Tank height h_wtes  [m]")
ax.set_ylabel("Heat loss  [kWh/h]")
ax.set_title("Absolute loss components  (SoC = 50 %)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)

ax = axes[1]
ax.plot(h_range, loss_frac, color="tab:red")
ax.set_xlabel("Tank height h_wtes  [m]")
ax.set_ylabel("Loss fraction  [% of C_WTES per hour]")
ax.set_title("Relative loss  (SoC = 50 %)")
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("wtes_loss_vs_height.png", dpi=150)
plt.show()
print("Saved: wtes_loss_vs_height.png")

# ==============================================================================
# 2. Loss rate as a function of SoC — for three representative tank sizes
# ==============================================================================
soc_range = np.linspace(0.05, 0.95, 200)

tank_configs = [
    ("0.5 m  ({:.1f} kWh)".format(kWh_per_m_wtes * 0.5),  0.5),
    ("2.0 m  ({:.1f} kWh)".format(kWh_per_m_wtes * 2.0),  2.0),
    ("10.0 m ({:.1f} kWh)".format(kWh_per_m_wtes * 10.0), 10.0),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("WTES loss rate vs SoC for different tank heights", fontsize=13)

for label, h in tank_configs:
    C = kWh_per_m_wtes * h
    Q = soc_range * C
    loss = beta_wtes * Q + gamma_wtes * C + loss_lids_wtes
    loss_pct = loss / C * 100
    axes[0].plot(soc_range * 100, loss_pct, label=label)
    axes[1].plot(soc_range * 100, loss, label=label)

axes[0].set_xlabel("State of Charge  [%]")
axes[0].set_ylabel("Loss fraction  [% of C_WTES per hour]")
axes[0].set_title("Relative loss")
axes[0].legend(title="Tank height")
axes[0].grid(True, alpha=0.4)

axes[1].set_xlabel("State of Charge  [%]")
axes[1].set_ylabel("Absolute heat loss  [kWh/h]")
axes[1].set_title("Absolute loss")
axes[1].legend(title="Tank height")
axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("wtes_loss_vs_soc.png", dpi=150)
plt.show()
print("Saved: wtes_loss_vs_soc.png")

# ==============================================================================
# 3. Self-discharge simulation — correct standby conditions
#    T_tank_init = 65 °C  (fully charged, matches datasheet test)
#    T_amb       = 20 °C  (laboratory / indoor ambient)
#    U_wall      = 0.4 W/(m²·K)  (validated against Vaillant VPS datasheet)
#
#    Physics: discrete hourly temperature decay
#       T(t) = T(t-1) - loss_hr(t) / (m·c / 3.6e6)
#       loss_hr(t) = U · A_tot · (T(t-1) - T_amb) / 1000   [kWh/h]
#    Analytical check: τ = m·c / (U·A_tot)  [hours]
# ==============================================================================

T_init_sim  = 65.0    # °C  — full tank at datasheet test temperature
T_amb_sim   = 20.0    # °C  — ambient
U_sim       = 0.4     # W/(m²·K)  — validated U
T_sim_days  = 30      # simulate 30 days
T_sim       = 24 * T_sim_days

# Simulate the three representative VPS tanks
sim_configs = [
    ("VPS 300/3",  300,  1.734),
    ("VPS 500/3",  500,  1.730),
    ("VPS 1000/3", 1000, 2.243),
]
sim_colors = ["#377eb8", "#e41a1c", "#4daf4a"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

hours = np.arange(T_sim + 1)

for (name, vol_L, h_m), col in zip(sim_configs, sim_colors):
    vol_m3 = vol_L / 1000.0
    d_m    = math.sqrt(4 * vol_m3 / (math.pi * h_m))
    A_lat  = math.pi * d_m * h_m
    A_lids = 2 * math.pi * (d_m / 2) ** 2
    A_tot  = A_lat + A_lids
    mass   = rho_water * vol_m3                       # kg
    mC_kWh = mass * c_water / 3.6e6                   # kWh/K  (thermal mass)
    tau_hr = mC_kWh * 1000 / (U_sim * A_tot)         # time constant [h]

    # Hourly temperature trace
    T_trace = np.zeros(T_sim + 1)
    T_trace[0] = T_init_sim
    loss_trace  = np.zeros(T_sim + 1)

    for t in range(1, T_sim + 1):
        loss_hr       = U_sim * A_tot * (T_trace[t - 1] - T_amb_sim) / 1000.0   # kWh/h
        loss_trace[t] = loss_hr
        dT            = loss_hr / mC_kWh                                          # °C drop
        T_trace[t]    = max(T_trace[t - 1] - dT, T_amb_sim)

    print(f"{name}: tau = {tau_hr:.1f} h = {tau_hr/24:.1f} days  |  "
          f"Standby @ t=0: {loss_trace[1]*24:.2f} kWh/24h  (datasheet: "
          f"< {dict(zip([c[0] for c in sim_configs], [1.7,2.0,2.5]))[name]} kWh/24h)")

    axes[0].plot(hours / 24, T_trace, color=col,
                 label=f"{name}  (tau = {tau_hr/24:.0f} d)")
    axes[1].plot(hours[1:] / 24, loss_trace[1:] * 24, color=col,
                 label=name)

# Panel 0 — temperature decay
axes[0].axhline(T_amb_sim, color="gray", linestyle=":", linewidth=0.9, label=f"T_amb = {T_amb_sim}°C")
axes[0].set_xlabel("Time  [days]")
axes[0].set_ylabel("Tank temperature  [°C]")
axes[0].set_title("Temperature decay (idle tank, no draw)")
axes[0].legend(fontsize=8)
axes[0].grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

# Panel 1 — daily standby loss
axes[1].set_xlabel("Time  [days]")
axes[1].set_ylabel("Standby loss  [kWh / 24 h]")
axes[1].set_title("Daily standby heat loss")
axes[1].legend(fontsize=8)
axes[1].grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

plt.tight_layout()
plt.savefig("wtes_self_discharge.png", dpi=150)
plt.show()
print("Saved: wtes_self_discharge.png")

# ==============================================================================
# 4. Numerical summary table
# ==============================================================================
print("\n=== Annual loss summary (idle tank, full year) ===")
print(f"{'h_wtes [m]':>12} {'C [kWh]':>10} {'Annual loss [kWh]':>20} {'Loss / C [%/yr]':>18}")
for h in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    C = kWh_per_m_wtes * h
    Q = 0.5 * C  # average SoC assumption
    loss_hr = beta_wtes * Q + gamma_wtes * C + loss_lids_wtes
    loss_yr = loss_hr * 8760
    print(f"{h:>12.1f} {C:>10.2f} {loss_yr:>20.1f} {loss_yr/C*100:>18.1f}")

# ==============================================================================
# 5. Validation against Vaillant VPS datasheet (standby loss kWh/24 h)
#    Source: Vaillant uniSTOR VPS product datasheet
#    Test condition (EN 12977-3): T_tank = 65 °C, T_ambient = 20 °C  → ΔT = 45 K
# ==============================================================================

# --- Datasheet values ---
vps_models   = ["VPS 300/3", "VPS 500/3", "VPS 800/3", "VPS 1000/3", "VPS 1500/3", "VPS 2000/3"]
vps_vol_L    = [300,  500,  800,  1000, 1500, 2000]   # litres
vps_h_mm     = [1734, 1730, 1870, 2243, 2253, 2394]   # tilt height ≈ standing height [mm]
vps_standby  = [1.7,  2.0,  2.4,  2.5,  2.9,  3.3]   # kWh/24 h (upper limit from datasheet)

# --- Test conditions ---
T_tank_test  = 65.0   # °C (EN 12977-3 standby test temperature)
T_amb_test   = 20.0   # °C (laboratory ambient)
dT_test      = T_tank_test - T_amb_test   # 45 K

# --- Derive geometry for each tank ---
# Volume (m³) = π*(d/2)² * h  →  d = sqrt(4*V / (π*h))
vps_vol_m3 = [v / 1000.0 for v in vps_vol_L]
vps_h_m    = [h / 1000.0 for h in vps_h_mm]
vps_d_m    = [math.sqrt(4 * v / (math.pi * h)) for v, h in zip(vps_vol_m3, vps_h_m)]

# Lateral wall area and lid area per tank
vps_A_lat  = [math.pi * d * h for d, h in zip(vps_d_m, vps_h_m)]
vps_A_lids = [2 * math.pi * (d / 2) ** 2 for d in vps_d_m]
vps_A_tot  = [al + ac for al, ac in zip(vps_A_lat, vps_A_lids)]

# --- Model prediction with current U_wall_wtes ---
# Standby = tank fully charged, no in/out, T_tank constant at T_tank_test
# Heat loss [kWh/24h] = U * A_total * ΔT * 24h / 1000  (W → kWh)
vps_model_24h = [U_wall_wtes * A * dT_test * 24 / 1000.0 for A in vps_A_tot]

# --- Back-calculate implied U from datasheet ---
vps_U_implied = [q_ds * 1000.0 / (A * dT_test * 24)
                 for q_ds, A in zip(vps_standby, vps_A_tot)]

# --- Print comparison table ---
print("\n=== Validation vs Vaillant VPS datasheet (standby kWh/24 h) ===")
print(f"  Test condition: T_tank = {T_tank_test}°C,  T_amb = {T_amb_test}°C,  ΔT = {dT_test} K")
print(f"  Model U_wall = {U_wall_wtes} W/(m²·K)\n")
hdr = (f"{'Model':<12} {'Vol[L]':>7} {'h[m]':>6} {'d[m]':>6} "
       f"{'A_tot[m²]':>10} {'Datasheet':>11} {'Model':>8} {'Ratio':>7} {'U_implied':>11}")
print(hdr)
print("-" * len(hdr))
for i, name in enumerate(vps_models):
    ratio = vps_model_24h[i] / vps_standby[i]
    print(f"{name:<12} {vps_vol_L[i]:>7} {vps_h_m[i]:>6.3f} {vps_d_m[i]:>6.3f} "
          f"{vps_A_tot[i]:>10.3f} {vps_standby[i]:>9.1f}   {vps_model_24h[i]:>7.2f} "
          f"  {ratio:>5.2f}x  {vps_U_implied[i]:>8.4f} W/(m²·K)")

print(f"\n  → Average implied U = {sum(vps_U_implied)/len(vps_U_implied):.4f} W/(m²·K)  "
      f"(model uses {U_wall_wtes} W/(m²·K))")

# --- Plot: datasheet vs model for a range of U values ---
U_values = [0.3, U_wall_wtes, 0.5]
colors   = ["tab:green", "tab:blue","tab:red"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Validation: WTES standby losses vs Vaillant VPS datasheet", fontsize=13)

# Left: absolute standby loss [kWh/24h]
ax = axes[0]
ax.bar(vps_models, vps_standby, color="silver", label="Datasheet (upper limit)", zorder=2)
for U_val, col in zip(U_values, colors):
    pred = [U_val * A * dT_test * 24 / 1000.0 for A in vps_A_tot]
    ax.plot(vps_models, pred, "o--", color=col, label=f"Model  U = {U_val} W/(m²·K)", zorder=3)
ax.set_ylabel("Standby loss  [kWh / 24 h]")
ax.set_title("Absolute standby loss")
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.4)

# Right: implied U-value per tank size
ax = axes[1]
ax.plot(vps_models, vps_U_implied, "s-", color="tab:purple", linewidth=2, label="Implied U from datasheet")
ax.axhline(U_wall_wtes, color="tab:orange", linestyle="--", label=f"Model U = {U_wall_wtes} W/(m²·K)")
ax.set_ylabel("U-value  [W / (m²·K)]")
ax.set_title("Back-calculated U from datasheet")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("wtes_validation_vps.png", dpi=150)
plt.show()
print("Saved: wtes_validation_vps.png")

# ==============================================================================
# 6. Viessmann SVPC validation
#    Source: Viessmann Vitocell 100-H SVPC datasheet
#    Geometry: diameter = 790 mm (without insulation), heights from table
#    Gross volumes used for water mass; outer surface area for losses
#    Two insulation variants: Standard and Efficient
#    Test condition assumed identical: T_tank = 65 °C, T_amb = 20 °C → ΔT = 45 K
# ==============================================================================

# --- SVPC data ---
svpc_models   = ["SVPC 600\nStd", "SVPC 600\nEff",
                 "SVPC 750\nStd", "SVPC 750\nEff",
                 "SVPC 910\nStd", "SVPC 910\nEff"]
svpc_vol_L    = [630.8, 630.8, 765.2, 765.2, 912.1, 912.1]   # gross volume [L]
svpc_d_mm     = [790,   790,   790,   790,   790,   790]      # diameter w/o insulation [mm]
svpc_h_mm     = [1535,  1535,  1815,  1815,  2120,  2120]     # height   w/o insulation [mm]
svpc_standby  = [2.68,  2.12,  2.74,  2.23,  2.81,  2.40]    # kWh/24 h from datasheet

# --- Derived geometry ---
svpc_vol_m3 = [v / 1000.0 for v in svpc_vol_L]
svpc_h_m    = [h / 1000.0 for h in svpc_h_mm]
svpc_d_m    = [d / 1000.0 for d in svpc_d_mm]
svpc_A_lat  = [math.pi * d * h for d, h in zip(svpc_d_m, svpc_h_m)]
svpc_A_lids = [2 * math.pi * (d / 2) ** 2 for d in svpc_d_m]
svpc_A_tot  = [al + ac for al, ac in zip(svpc_A_lat, svpc_A_lids)]

# --- Back-calculate implied U ---
svpc_U_implied = [q * 1000.0 / (A * dT_test * 24)
                  for q, A in zip(svpc_standby, svpc_A_tot)]

# --- Model prediction at U = 0.4 ---
svpc_model_24h = [0.4 * A * dT_test * 24 / 1000.0 for A in svpc_A_tot]

# --- Print SVPC comparison table ---
print("\n=== Validation vs Viessmann SVPC datasheet (standby kWh/24 h) ===")
print(f"  Test condition: T_tank = {T_tank_test}°C,  T_amb = {T_amb_test}°C,  ΔT = {dT_test} K\n")
hdr2 = (f"{'Model':<14} {'Vol[L]':>7} {'h[m]':>6} {'d[m]':>6} "
        f"{'A_tot[m²]':>10} {'Datasheet':>11} {'U=0.4':>8} {'U_implied':>11}")
print(hdr2)
print("-" * len(hdr2))
for i, name in enumerate(svpc_models):
    print(f"{name.replace(chr(10),' '):<14} {svpc_vol_L[i]:>7.1f} {svpc_h_m[i]:>6.3f} "
          f"{svpc_d_m[i]:>6.3f} {svpc_A_tot[i]:>10.3f} {svpc_standby[i]:>9.2f}   "
          f"{svpc_model_24h[i]:>7.2f}  {svpc_U_implied[i]:>8.4f} W/(m²·K)")
print(f"\n  → Average implied U (SVPC) = "
      f"{sum(svpc_U_implied)/len(svpc_U_implied):.4f} W/(m²·K)")

# ==============================================================================
# 7. Combined LaTeX-quality figure — both manufacturers, justification U = 0.4
# ==============================================================================
import matplotlib as mpl

mpl.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8.5,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   1.5,
})

U_candidates = [0.2, 0.3, 0.4, 0.6, 1.0]
line_styles  = [":",  "--", "-",  "-.", ":"]
line_colors  = ["#888888", "#4daf4a", "#e41a1c", "#377eb8", "#ff7f00"]
line_widths  = [1.2,  1.2,  2.2,  1.2,  1.2]

# Combined volume axis: VPS + SVPC (unique volumes only for model lines)
all_vols_sorted = sorted(set(vps_vol_L + [v for v in svpc_vol_L]))
vol_line = np.linspace(200, 2200, 300)

# Back-calculate A_tot as a function of volume using VPS geometry (d derived per tank)
# For model lines we need A(V): use VPS-derived relationship (d from sqrt formula)
# Use a representative average h/d ratio from both datasets combined
all_vol_m3  = [v/1000 for v in vps_vol_L] + svpc_vol_m3
all_h_m     = vps_h_m + svpc_h_m
all_d_m     = vps_d_m + svpc_d_m
all_A_tot   = vps_A_tot + svpc_A_tot

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))

# ── Left panel: standby loss vs volume ───────────────────────────────────────
ax = axes[0]

# Model lines using actual surface areas interpolated vs volume
for U_val, ls, col, lw in zip(U_candidates, line_styles, line_colors, line_widths):
    # fit a simple linear A_tot vs volume relationship from all data points
    vol_all_L = vps_vol_L + svpc_vol_L
    A_fit  = np.polyfit(vol_all_L, all_A_tot, 1)
    A_line = np.polyval(A_fit, vol_line)
    pred   = U_val * A_line * dT_test * 24 / 1000.0
    lbl    = f"$U = {U_val}$" + (" ← chosen" if U_val == 0.4 else "")
    ax.plot(vol_line, pred, linestyle=ls, color=col, linewidth=lw, label=lbl)

# Datasheet points — two manufacturers
ax.scatter(vps_vol_L, vps_standby,
           marker="v", s=50, color="black", zorder=5, label="Vaillant VPS")
ax.scatter([v for v in svpc_vol_L[::2]], [s for s in svpc_standby[::2]],
           marker="s", s=45, color="#377eb8", zorder=5, label="Viessmann SVPC (std)")
ax.scatter([v for v in svpc_vol_L[1::2]], [s for s in svpc_standby[1::2]],
           marker="^", s=45, color="#4daf4a", zorder=5, label="Viessmann SVPC (eff)")

ax.set_xlabel("Tank volume  [L]")
ax.set_ylabel("Standby loss  [kWh / 24 h]")
ax.set_title("(a)  Model vs. manufacturer data")
ax.legend(loc="lower right", framealpha=0.0, handlelength=1.8, ncol=1)
ax.set_xlim(200, 2200)
ax.set_ylim(0, None)
ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

# ── Right panel: implied U for both manufacturers ────────────────────────────
ax = axes[1]

ax.plot(vps_vol_L, vps_U_implied, "v-",
        color="black", linewidth=1.6, markersize=5, label="Vaillant VPS")
ax.plot(svpc_vol_L[::2], svpc_U_implied[::2], "s-",
        color="#377eb8", linewidth=1.6, markersize=5, label="SVPC standard")
ax.plot(svpc_vol_L[1::2], svpc_U_implied[1::2], "^-",
        color="#4daf4a", linewidth=1.6, markersize=5, label="SVPC efficient")

ax.axhline(0.4, color="#e41a1c", linewidth=2.0, linestyle="-",
           label="$U = 0.4$ W/(m²·K)  [chosen]")

# Shade the conservative margin (where chosen U is above implied → safe side)
all_vols_combined = vps_vol_L + svpc_vol_L[::2] + svpc_vol_L[1::2]
all_U_combined    = vps_U_implied + svpc_U_implied[::2] + svpc_U_implied[1::2]
for v, u in zip(all_vols_combined, all_U_combined):
    if u > 0.4:
        ax.annotate("", xy=(v, 0.4), xytext=(v, u),
                    arrowprops=dict(arrowstyle="-", color="#e41a1c",
                                   lw=0.8, linestyle="dashed"))

ax.set_xlabel("Tank volume  [L]")
ax.set_ylabel("Implied $U$-value  [W/(m²·K)]")
ax.set_title("(b)  Back-calculated $U$ from datasheets")
ax.set_xlim(200, 2200)
ax.set_ylim(0, 0.80)
ax.legend(loc="lower left", framealpha=0.0, handlelength=1.6)
ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

fig.tight_layout(pad=0.8)
fig.savefig("wtes_U_justification.pdf", bbox_inches="tight")
fig.savefig("wtes_U_justification.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: wtes_U_justification.pdf  +  wtes_U_justification.png  (combined, both manufacturers)")
