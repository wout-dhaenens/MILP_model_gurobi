import pulp
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# --- 1. GLOBAL PARAMETERS & PHYSICAL CONSTANTS ---
# ==============================================================================
T_HOURS = 24
time = np.arange(T_HOURS)
dt_sec = 3600 

d = 1.0         # Diameter [m]
rho = 971.8     
c_water = 4190  
temp_h, temp_c, temp_env = 70, 50, 10
u_value = 1.0
eta_in, eta_out = 0.95, 0.95
COP_HP = 3.5

delta_T_HC = temp_h - temp_c
delta_T_C0 = temp_c - temp_env
delta_T_H0 = temp_h - temp_env

# --- Linear Coefficient Calculation ---
kWh_per_m = (np.pi * (d**2 / 4) * rho * c_water * delta_T_HC) / 3.6e6
beta = u_value * (4 / (d * rho * c_water)) * dt_sec
gamma = u_value * (4 / (d * rho * c_water * delta_T_HC)) * delta_T_C0 * dt_sec
loss_lids_kwh = (u_value * (2 * np.pi * (d**2 / 4)) * (delta_T_H0 + delta_T_C0) * dt_sec / 2) / 3.6e6

# Costs
CAPEX_HP_ANNUAL = 150.0
CAPEX_TES_METER_ANNUAL = 100.0 

# ==============================================================================
# --- 2. OPTIMIZATION & PLOTTING ---
# ==============================================================================

def run_tes_height_optimization(pv_profile, price_scenario, name):
    prob = pulp.LpProblem(f"TES_Height_Optimization_{name}", pulp.LpMinimize)

    # Decision Variables
    h_tank = pulp.LpVariable('h_tank', lowBound=0.5, upBound=10.0) 
    C_HP = pulp.LpVariable('C_HP', lowBound=0)
    P_buy = pulp.LpVariable.dicts('P_buy', time, lowBound=0)
    P_hp_elec = pulp.LpVariable.dicts('P_hp_elec', time, lowBound=0)
    Q_tes = pulp.LpVariable.dicts('Q_tes', time, lowBound=0) 

    C_TES_expr = h_tank * kWh_per_m

    # Objective
    inv_cost = (CAPEX_HP_ANNUAL * C_HP) + (CAPEX_TES_METER_ANNUAL * h_tank)
    op_cost = pulp.lpSum([price_scenario[t] * P_buy[t] for t in time])
    prob += inv_cost + 365 * op_cost

    # Constraints
    for t in time:
        prob += P_buy[t] + pv_profile[t] >= P_hp_elec[t]
        q_hp_thermal = P_hp_elec[t] * COP_HP
        prob += q_hp_thermal <= C_HP
        
        prev = T_HOURS - 1 if t == 0 else t - 1
        loss_t = (beta * Q_tes[prev]) + (gamma * C_TES_expr) + loss_lids_kwh
        gain_t = (q_hp_thermal * eta_in) - (P_THERMAL_LOAD[t] / eta_out)
        
        prob += Q_tes[t] == Q_tes[prev] - loss_t + gain_t
        prob += Q_tes[t] <= 0.95 * C_TES_expr
        prob += Q_tes[t] >= 0.05 * C_TES_expr

    prob += Q_tes[T_HOURS-1] == Q_tes[0]
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        opt_h = pulp.value(h_tank)
        opt_cap = opt_h * kWh_per_m
        res_q = np.array([pulp.value(Q_tes[t]) for t in time])
        res_hp_th = np.array([pulp.value(P_hp_elec[t]) * COP_HP for t in time])
        res_grid = np.array([pulp.value(P_buy[t]) for t in time])
        res_losses = np.array([(beta * res_q[T_HOURS-1 if t==0 else t-1] + gamma * opt_cap + loss_lids_kwh) for t in time])

        # --- Plotting ---
        fig, ax = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
        
        # Panel 1: Thermal Balance
        ax[0].step(time, P_THERMAL_LOAD, label='Thermal Load', where='post', lw=2, color='black', ls='--')
        ax[0].step(time, res_hp_th, label='HP Production', where='post', lw=2, color='red', alpha=0.7)
        ax[0].set_ylabel("Power [kW_th]")
        ax[0].legend(loc='upper right')
        ax[0].grid(True, alpha=0.3)
        ax[0].set_title(f"Optimized Height: {opt_h:.2f} m | Capacity: {opt_cap:.1f} kWh | HP: {pulp.value(C_HP):.1f} kW_th")

        # Panel 2: Storage & Losses
        ax1_twin = ax[1].twinx()
        l1 = ax[1].plot(time, res_q, label='Storage Energy', lw=3, color='blue')
        ax[1].set_ylabel("Stored Energy [kWh]", color='blue')
        l2 = ax1_twin.plot(time, res_losses, label='Thermal Losses', lw=2, color='darkred', ls='-')
        ax1_twin.set_ylabel("Losses [kW]", color='darkred')
        ax[1].legend(l1+l2, [l.get_label() for l in l1+l2], loc='upper right')
        ax[1].grid(True, alpha=0.3)

        # Panel 3: Electricity & Market
        ax2_twin = ax[2].twinx()
        ax[2].bar(time, res_grid, label='Grid Buy', color='green', alpha=0.3)
        ax[2].plot(time, pv_profile, label='PV Solar', color='orange', lw=2)
        ax2_twin.step(time, price_scenario, label='Price', where='post', color='purple')
        ax[2].set_ylabel("Electric Power [kW_e]")
        ax2_twin.set_ylabel("Price [€/kWh]", color='purple')
        ax[2].set_xlabel("Hour of Day")
        ax[2].grid(True, alpha=0.3)

        plt.suptitle(f"TES Height Optimization: {name}", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
    else:
        print("Optimization failed.")

# --- Execution ---
P_THERMAL_LOAD = np.array([2.0, 2.0, 2.0, 2.5, 3.0, 5.0, 8.0, 10.0, 8.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 8.0, 10.0, 9.0, 7.0, 5.0, 3.0, 2.0, 2.0])
dummy_pv = np.array([max(0, 8 * np.sin(np.pi * (t-6)/12)) if 6 <= t <= 18 else 0 for t in range(24)])
dummy_prices = np.array([0.1, 0.1, 0.1, 0.1, 0.12, 0.15, 0.4, 0.5, 0.4, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.15, 0.3, 0.5, 0.6, 0.4, 0.2, 0.1, 0.1, 0.1])

run_tes_height_optimization(dummy_pv, dummy_prices, "Scenario A")