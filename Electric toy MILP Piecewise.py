import pulp
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

# ==============================================================================
# --- 1. GLOBAL PARAMETERS ---
# ==============================================================================

time = np.arange(24)
T = len(time)
dt = 1.0 # hours
BIG_M = 4e10 

# Investment data (15y lifetime, 5% interest for ANF calculation)
LIFETIME = 15
INTEREST_RATE = 0.05
ANF = (INTEREST_RATE * (1 + INTEREST_RATE)**LIFETIME) / ((1 + INTEREST_RATE)**LIFETIME - 1) # ~0.09634
CAPEX_PV_ANNUAL = 78.05 

# Raw BYD Battery Data (Capacity in kWhe, Price in €)
RAW_BATTERY_DATA = {
    'HVS': [
        (5.12, 3813), (7.68, 5379), (10.24, 6945), (12.8, 8512)
    ],
    'HVM': [
        (8.3, 5429), (11.04, 7012), (13.8, 8644), (16.56, 10227), (19.32, 11810), (22.08, 13392)
    ],
    'LVS': [
        (4, 2756), (8, 4886), (12, 6974), (16, 9092), (20, 11130), (24, 13236)
    ]
}

# Battery parameters 
ETA_BAT_CH = 0.95
ETA_BAT_DIS = 0.95
P_BAT_POWER_RATIO_FIXED = 1.0 

# Demand and Solar Profiles
P_LOAD = 10*np.array([0.5, 0.5, 0.5, 0.6, 0.8, 1.2, 2.5, 3.5, 4.5, 5.0, 5.5, 5.8, 6.0, 5.9, 5.5, 5.2, 4.8, 3.5, 1.5, 1.0, 0.7, 0.6, 0.5, 0.5])
P_LOAD[P_LOAD < 0] = 0
I_SOLAR = np.array([max(0, np.sin(2*np.pi*(t-6)/24)) for t in time])

# ==============================================================================
# --- 2. DATA PREPARATION (Piecewise Function and Price Loading) ---
# ==============================================================================

def calculate_piecewise_data(raw_data, anf):
    """
    Calculates the Piecewise Linear (PWL) parameters (Points) for each battery family.
    """
    pwl_data = {}
    for tech, points in raw_data.items():
        cap_points = np.array([p[0] for p in points])
        price_points = np.array([p[1] for p in points])
        
        # Calculate ANNUALIZED COST POINTS (€/a)
        annual_cost_points = price_points * anf 
        
        pwl_data[tech] = {
            'CAP_POINTS': cap_points,
            'ANNUAL_COST_POINTS': annual_cost_points,
            'NUM_SEGMENTS': len(cap_points)
        }
    return pwl_data

def load_price_data(filename="load_data.csv", time_horizon=T):
    """Loads and processes the electricity price data from file."""
    # Placeholder/Fallback array (Scaled to 0.15 to 0.25 €/kWh)
    P_price_base_fallback = np.array([0.15] * 8 + [0.25] * 8 + [0.20] * 8)
    
    try:
        try: price_df = pd.read_csv(filename, sep=';', encoding='windows-1252')
        except: price_df = pd.read_csv(filename, sep=',', encoding='windows-1252')
            
        if price_df.shape[1] >= 2:
            price_df = price_df.iloc[:, 0:2]
            price_df.columns = ['Date', 'Euro']
        else: raise ValueError("CSV must contain at least two columns.")

        price_df['Euro_Cleaned'] = (
            price_df['Euro']
            .astype(str)
            .str.replace(r'[^\d,]', '', regex=True)
            .str.replace(',', '.', regex=False)
            .astype(float)
        )
        
        P_price_data = (price_df['Euro_Cleaned'].to_numpy() / 1000)
        P_price_base = P_price_data[::-1]
        
        if len(P_price_base) < time_horizon:
            P_price_base = np.pad(P_price_base, (0, time_horizon - len(P_price_base)), mode='edge')
        elif len(P_price_base) > time_horizon:
             P_price_base = P_price_base[:time_horizon]
        
        return {"Base": P_price_base}
        
    except Exception as e:
        print(f"Error processing '{filename}' or file not found: {e}. Using placeholder price (Base).")
        return {"Base": P_price_base_fallback}

def define_scenarios(base_prices, pwl_data):
    """
    Defines scenarios based on different battery technologies.
    """
    scenarios = {}
    for tech in pwl_data.keys():
        scenarios[f"{tech} Technology"] = {
            'price_array': base_prices['Base'],
            'tech_data': pwl_data[tech]
        }
    return scenarios


# ==============================================================================
# --- 4. PLOTTING FUNCTIONS ---
# ==============================================================================

def plot_piecewise_capex(C_bat_opt, annual_cost_opt, tech_data, scenario_name):
    """
    Plots the piecewise linear CAPEX function and indicates the optimal solution.
    """
    CAP_POINTS = tech_data['CAP_POINTS']
    COST_POINTS = tech_data['ANNUAL_COST_POINTS']
    
    plt.figure(figsize=(8, 6))
    
    # 1. Plot the linear segments (the PWL function)
    plt.plot(CAP_POINTS, COST_POINTS, 'b-', label='PWL Cost Function (Segments)', alpha=0.6)
    
    # 2. Plot the discrete data points (nodes)
    plt.plot(CAP_POINTS, COST_POINTS, 'bo', label='Discrete Module Points')
    
    # 3. Plot the optimal solution found by the MILP
    # To get the interpolated cost at the optimal capacity, we use the fact that
    # the solver calculated the objective component 'annualized_capex_bat'.
    plt.plot(C_bat_opt, annual_cost_opt, 'go', markersize=10, 
             label=f'Optimal Solution ({C_bat_opt:.2f} kWh)')
    
    plt.title(f'Piecewise CAPEX Model - {scenario_name}')
    plt.xlabel('Storage Capacity [kWh]')
    plt.ylabel('Annualized CAPEX [€/a]')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

# ==============================================================================
# --- 5. CORE OPTIMIZATION FUNCTION ---
# ==============================================================================

def run_scenario_optimization(scenario_data, scenario_name):
    """
    Builds, solves the MILP model for a given battery technology, and plots results.
    """
    
    P_price_scenario = scenario_data['price_array']
    tech_data = scenario_data['tech_data']
    
    CAP_POINTS = tech_data['CAP_POINTS']
    COST_POINTS = tech_data['ANNUAL_COST_POINTS']
    NUM_SEGMENTS = tech_data['NUM_SEGMENTS']
    
    # --- A. Initialize Problem ---
    prob = pulp.LpProblem(f"Energy_System_MILP_{scenario_name}", pulp.LpMinimize)
    
    # --- B. Define Variables ---
    C_PV = pulp.LpVariable('C_PV', lowBound=0)
    y_PV = pulp.LpVariable('y_PV', cat='Binary')
    
    # PWL VARIABLES
    C_bat_pwl = pulp.LpVariable('C_bat_pwl', lowBound=CAP_POINTS[0], upBound=CAP_POINTS[-1]) 
    annualized_capex_bat_var = pulp.LpVariable('Annual_CAPEX_Bat')
    
    lambda_vars = pulp.LpVariable.dicts("Lambda", range(NUM_SEGMENTS), lowBound=0, upBound=1)
    
    # Operational Variables (Flows)
    P_buy = pulp.LpVariable.dicts('P_buy', time, lowBound=0) 
    P_sell = pulp.LpVariable.dicts('P_sell', time, lowBound=0)
    P_bat_ch = pulp.LpVariable.dicts('P_bat_ch', time, lowBound=0)
    P_bat_dis = pulp.LpVariable.dicts('P_bat_dis', time, lowBound=0)
    SoC_bat = pulp.LpVariable.dicts('SoC_bat', time, lowBound=0)
    
    # --- C. Set Objective (Cost Minimization) ---
    
    # 1. Investment Costs (Annualized)
    annualized_inv = CAPEX_PV_ANNUAL * C_PV + annualized_capex_bat_var # Use the PWL cost variable
    
    # 2. Operational Costs & Revenue
    operational_cost_buy = pulp.lpSum([P_price_scenario[t] * P_buy[t] * dt for t in time])
    P_feed_in_price = P_price_scenario * 0.1 
    operational_revenue_sell = pulp.lpSum([P_feed_in_price[t] * P_sell[t] * dt for t in time])
    
    # Total Objective
    prob += annualized_inv + 365 * (operational_cost_buy - operational_revenue_sell), "Total_Annualized_Cost"
    
    # --- D. Add Constraints ---
    
    # D.1. Piecewise Linear Constraints (SOS2)
    
    # 1. Battery Capacity (C_bat_pwl) is defined by the linear combination of capacities at the nodes
    prob += C_bat_pwl == pulp.lpSum([lambda_vars[j] * CAP_POINTS[j] for j in range(NUM_SEGMENTS)]), "PWL_Capacity_Interpolation"
    
    # 2. Annualized CAPEX is defined by the linear combination of costs at the nodes
    prob += annualized_capex_bat_var == pulp.lpSum([lambda_vars[j] * COST_POINTS[j] for j in range(NUM_SEGMENTS)]), "PWL_Cost_Interpolation"

    # 3. Sum of lambda weights must equal 1
    prob += pulp.lpSum([lambda_vars[j] for j in range(NUM_SEGMENTS)]) == 1, "PWL_Sum_of_Lambdas"

    # 4. SOS2 Constraint (At most two adjacent lambdas can be non-zero)
    prob.sos2["PWL_SOS2_Constraint"] = [lambda_vars[j] for j in range(NUM_SEGMENTS)]

    # D.2. Operational Constraints

    # Capacity Coupling (PV only)
    prob += C_PV <= BIG_M * y_PV, "Capacity_Coupling_PV"

    # Energy Balance (Power In = Power Out)
    for t in time:
        PV_prod_t = I_SOLAR[t] * C_PV
        inflow = PV_prod_t + P_bat_dis[t] + P_buy[t]
        outflow = P_LOAD[t] + P_bat_ch[t] + P_sell[t]
        prob += inflow == outflow, f"Energy_Balance_t{t}"
        
        # Battery Operational Limits (using the PWL-derived capacity C_bat_pwl)
        prob += P_bat_ch[t] <= P_BAT_POWER_RATIO_FIXED * C_bat_pwl, f"Bat_Max_Charge_t{t}"
        prob += P_bat_dis[t] <= P_BAT_POWER_RATIO_FIXED * C_bat_pwl, f"Bat_Max_Discharge_t{t}"
        prob += SoC_bat[t] <= C_bat_pwl, f"SoC_Max_Capacity_t{t}"

    # Battery SoC Dynamics (Circular)
    T_max = len(time)
    for t in time:
        prev = T_max - 1 if t == 0 else t - 1
        prob += (
            SoC_bat[t] == SoC_bat[prev] + 
            (ETA_BAT_CH * P_bat_ch[t] - (1/ETA_BAT_DIS) * P_bat_dis[t]) * dt
        ), f"SoC_Dynamics_t{t}"
        
    prob += SoC_bat[T_max - 1] == SoC_bat[0], "SoC_Circular_Constraint"
    
    # --- E. Solve ---
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]

    # --- F. Extract Results ---
    if status == "Optimal":
        C_PV_opt = pulp.value(C_PV)
        C_bat_opt = pulp.value(C_bat_pwl) 
        annual_cost_opt = pulp.value(annualized_capex_bat_var)
        
        results = {
            'C_PV': C_PV_opt,
            'C_bat': C_bat_opt,
            'Annual_Cost_Bat': annual_cost_opt, # Pass this to plot the dot
            'Objective': pulp.value(prob.objective),
            'SoC_bat': np.array([pulp.value(SoC_bat[t]) for t in time]),
            'P_buy': np.array([pulp.value(P_buy[t]) for t in time]),
            'P_sell': np.array([pulp.value(P_sell[t]) for t in time]),
            'P_bat_ch': np.array([pulp.value(P_bat_ch[t]) for t in time]),
            'P_bat_dis': np.array([pulp.value(P_bat_dis[t]) for t in time]),
            'PV_prod': I_SOLAR * C_PV_opt,
            'P_load': P_LOAD
        }
        
        # Plot the PWL cost curve and optimal solution
        plot_piecewise_capex(C_bat_opt, annual_cost_opt, tech_data, scenario_name)
        
    else:
        print(f"ERROR: Scenario '{scenario_name}' failed to reach optimal status: {status}")
        return

    # --- G. Plotting operational results ---
    C_PV_opt = results['C_PV']
    C_bat_opt = results['C_bat']
    P_price_scenario = scenario_data['price_array']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    plt.suptitle(f"Scenario: {scenario_name} | Cost: {results['Objective']:.2f} €/a \n C_PV={C_PV_opt:.2f} kWp, C_bat={C_bat_opt:.2f} kWh", fontsize=12)

    # Subplot 1: Power Flows (kW)
    axes[0].plot(time, results['P_load'], label="Electrical Load (P_load)", color='black', linestyle='--')
    axes[0].plot(time, results['PV_prod'], label="PV Production (P_PV)", color='orange', linewidth=2)
    P_grid_net = results['P_buy'] - results['P_sell']
    axes[0].plot(time, P_grid_net, label="Grid Net Flow (Buy - Sell)", color='red', linewidth=2)
    P_bat_net = results['P_bat_dis'] - results['P_bat_ch']
    axes[0].plot(time, P_bat_net, label="Battery Net Flow (Dis - Ch)", color='green', linewidth=2)
    
    axes[0].set_ylabel("Power [kW]", fontsize=12)
    axes[0].set_title("Power Flows", fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='upper left', ncol=2, fontsize=9)
    axes[0].axhline(0, color='gray', linewidth=0.5, linestyle='-')

    # Subplot 2: Battery State (kWh)
    axes[1].plot(time, results['SoC_bat'], label="Battery SoC", color='blue', linewidth=2)
    axes[1].axhline(y=C_bat_opt, color='blue', linestyle='--', alpha=0.5, label=f"Capacity ({C_bat_opt:.2f} kWh)")
    axes[1].set_ylabel("Energy [kWh]", fontsize=12)
    axes[1].set_title("Battery State of Charge", fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='lower left', fontsize=9)
    
    # Subplot 3: Price Profile
    P_feed_in_price = P_price_scenario * 0.1 # Recalculate for plotting
    axes[2].plot(time, P_price_scenario, label="Electricity Price (Buy)", color='purple', linewidth=2)
    axes[2].plot(time, P_feed_in_price, label="Feed-in Price (Sell)", color='gray', linestyle='--')
    axes[2].set_ylabel("Price [€/kWh]", fontsize=12)
    axes[2].set_xlabel("Time [h]", fontsize=12)
    axes[2].set_title("Price Profile", fontsize=14)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(loc='upper left', fontsize=9)
    axes[2].set_ylim(0, np.max(P_price_scenario) * 1.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==============================================================================
# --- 6. MAIN EXECUTION BLOCK (Scenario Runner) ---
# ==============================================================================

if __name__ == "__main__":
    # 1. Pre-process CAPEX data and load prices
    PWL_DATA = calculate_piecewise_data(RAW_BATTERY_DATA, ANF)
    base_prices = load_price_data()
    
    # 2. Define scenarios based on battery technology CAPEX
    all_scenarios = define_scenarios(base_prices, PWL_DATA)

    # 3. Iterate through all defined scenarios
    print("Starting piecewise CAPEX battery technology simulation...")
    for name, data in all_scenarios.items():
        print(f"\n--- Running Scenario: {name} ---")
        run_scenario_optimization(data, name)
        
    print("\nScenario simulation complete.")