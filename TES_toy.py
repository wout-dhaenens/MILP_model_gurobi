import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parameters & Pre-calculations ---
d = 1.0             # Diameter [m]
h = 5.0             # Height [m]
rho = 971.8         # Density [kg/m3]
c = 4190            # Heat capacity [J/(kg*K)]
temp_h = 70         # Hot zone [°C]
temp_c = 50         # Cold zone [°C]
temp_env = 20       # Ambient [°C]
u_value = 0.5         # Thermal transmittance [W/(m2*K)]
dt = 3600           # Time step [s] (1 hour)

delta_T_HC, delta_T_C0, delta_T_H0 = temp_h - temp_c, temp_c - temp_env, temp_h - temp_env
volume = np.pi * (d**2 / 4) * h
Q_N = volume * rho * c * delta_T_HC
print(Q_N/3600/1000)
# Model Coefficients
beta = u_value * (4 / (d * rho * c)) * dt
gamma = u_value * (4 / (d * rho * c * delta_T_HC)) * delta_T_C0 * dt
delta = u_value * (np.pi * d**2 / 4) * (delta_T_H0 + delta_T_C0) * dt

# --- 2. Simulation ---
durations = [12, 48, 6, 8] # [Charge, Standby, Re-charge, Discharge]
steps = sum(durations)
Q = np.zeros(steps)
losses_kw = np.zeros(steps)
Q[0] = Q_N * 0.1  

P_in, P_out, eta = 0.15*(Q_N/3600), 0.25*(Q_N/3600), 0.95

for t in range(1, steps):
    # Operating Modes
    if t < 12: q_in, q_out = P_in, 0
    elif 12 <= t < 60: q_in, q_out = 0, 0
    elif 60 <= t < 66: q_in, q_out = P_in, 0
    else: q_in, q_out = 0, P_out
    
    # Calculate Instantaneous Loss
    current_loss_j = (beta * Q[t-1]) + (gamma * Q_N) + delta
    losses_kw[t] = current_loss_j / dt / 1000
    
    # Update state
    term_gain = (q_in * eta * dt) - (q_out / eta * dt)
    Q[t] = max(min(Q[t-1] - current_loss_j + term_gain, Q_N * 0.95), Q_N * 0.05)

Q_mwh = Q / 3.6e9

# --- 3. Combined Visualization ---
fig, ax1 = plt.subplots(figsize=(12, 7))

# Left Axis: MWh
ax1.set_xlabel('Time [Hours]', fontsize=12)
ax1.set_ylabel('Stored Energy [MWh]', color='tab:blue', fontweight='bold')
line1 = ax1.plot(range(steps), Q_mwh, color='tab:blue', lw=3, label='Energy Content (MWh)')
ax1.fill_between(range(steps), Q_mwh, alpha=0.1, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Right Axis: kW
ax2 = ax1.twinx()
ax2.set_ylabel('Heat Losses [kW]', color='tab:red', fontweight='bold')
line2 = ax2.plot(range(steps), losses_kw, color='tab:red', ls='--', lw=2, label='Heat Losses (kW)')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Phase Overlays
colors, labels = ['#e1f5fe', '#fff9c4', '#e1f5fe', '#ffebee'], ['Charge', 'Standby', 'Re-charge', 'Discharge']
start_h = 0
for i, d_h in enumerate(durations):
    ax1.axvspan(start_h, start_h + d_h, color=colors[i], alpha=0.3)
    ax1.text(start_h + d_h/2, max(Q_mwh)*1.02, labels[i], ha='center', fontweight='bold')
    start_h += d_h

plt.title("TES Performance: Energy Storage (MWh) vs. Thermal Losses (kW)", fontsize=14)
ax1.legend(line1 + line2, [l.get_label() for l in line1+line2], loc='center right')
ax1.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()