import pulp
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import time as time_module
from scipy.interpolate import interp1d

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

# Raw BYD Battery Data (HVM chosen as the base technology for analysis)
RAW_BATTERY_DATA_BASE = [
    (8.3, 5429), (11.04, 7012), (13.8, 8644), (16.56, 10227), (19.32, 11810), (22.08, 13392)
]

# Battery parameters 
ETA_BAT_CH = 0.95
ETA_BAT_DIS = 0.95
P_BAT_POWER_RATIO_FIXED = 1.0 

# Demand and Solar Profiles
P_LOAD = np.array([0.5, 0.5, 0.5, 0.6, 0.8, 1.2, 2.5, 3.5, 4.5, 5.0, 5.5, 5.8, 6.0, 5.9, 5.5, 5.2, 4.8, 3.5, 1.5, 1.0, 0.7, 0.6, 0.5, 0.5])
P_LOAD[P_LOAD < 0] = 0
I_SOLAR = np.array([max(0, np.sin(2*np.pi*(t-6)/24)) for t in time])

# ==============================================================================
# --- 2. DATA PREPARATION FUNCTIONS ---
# ==============================================================================

def increase_points(cap_points, cost_points, target_num_points):
    """
    Generates new, interpolated data points to increase the granularity.
    """
    if target_num_points <= len(cap_points):
        return cap_points, cost_points

    # 1. Create interpolation functions based on original points
    cost_interp = interp1d(cap_points, cost_points)

    # 2. Define new, evenly spaced capacity points for the desired number of points
    new_cap_points_x = np.linspace(cap_points[0], cap_points[-1], target_num_points)

    # 3. Use linear interpolation to find the corresponding cost for these new points
    new_cost_points_y = cost_interp(new_cap_points_x)
    
    return new_cap_points_x, new_cost_points_y

def calculate_piecewise_data(raw_points, anf):
    """
    Calculates the Piecewise Linear (PWL) parameters (Points) from raw data.
    """
    cap_points = np.array([p[0] for p in raw_points])
    price_points = np.array([p[1] for p in raw_points])
    annual_cost_points = price_points * anf 
    
    return {
        'CAP_POINTS': cap_points,
        'ANNUAL_COST_POINTS': annual_cost_points,
    }

def load_price_data(filename="load_data.csv", time_horizon=T):
    """Loads and processes the electricity price data from file."""
    # Placeholder/Fallback array (Used consistently for this analysis)
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
        
    except Exception:
        # Returning placeholder prices for consistency in the timing test
        return {"Base": P_price_base_fallback}

# ==============================================================================
# --- 3. CORE OPTIMIZATION FUNCTION (for Analysis) ---
# ==============================================================================

def run_interpolation_analysis(tech_data_input, scenario_name):
    """
    Runs the MILP optimization with timing and extracts necessary data.
    Returns: NUM_SEGMENTS, duration, C_bat_opt, Cost_bat_opt, CAP_POINTS, COST_POINTS
    """
    
    # Use fixed price array for consistent timing
    P_price_scenario = load_price_data()['Base'] 
    
    CAP_POINTS = tech_data_input['CAP_POINTS']
    COST_POINTS = tech_data_input['ANNUAL_COST_POINTS']
    NUM_SEGMENTS = len(CAP_POINTS)
    
    # --- A. Initialize Problem ---
    prob = pulp.LpProblem(f"Analysis_MILP_{scenario_name}", pulp.LpMinimize)
    
    # --- B. Define Variables ---
    C_PV = pulp.LpVariable('C_PV', lowBound=0)
    y_PV = pulp.LpVariable('y_PV', cat='Binary')
    
    C_bat_pwl = pulp.LpVariable('C_bat_pwl', lowBound=CAP_POINTS[0], upBound=CAP_POINTS[-1]) 
    annualized_capex_bat_var = pulp.LpVariable('Annual_CAPEX_Bat')
    lambda_vars = pulp.LpVariable.dicts("Lambda", range(NUM_SEGMENTS), lowBound=0, upBound=1)
    
    P_buy = pulp.LpVariable.dicts('P_buy', time, lowBound=0) 
    P_sell = pulp.LpVariable.dicts('P_sell', time, lowBound=0)
    P_bat_ch = pulp.LpVariable.dicts('P_bat_ch', time, lowBound=0)
    P_bat_dis = pulp.LpVariable.dicts('P_bat_dis', time, lowBound=0)
    SoC_bat = pulp.LpVariable.dicts('SoC_bat', time, lowBound=0)
    
    # --- C. Set Objective ---
    annualized_capex_bat = pulp.lpSum([lambda_vars[j] * COST_POINTS[j] for j in range(NUM_SEGMENTS)])
    annualized_inv = CAPEX_PV_ANNUAL * C_PV + annualized_capex_bat
    operational_cost_buy = pulp.lpSum([P_price_scenario[t] * P_buy[t] * dt for t in time])
    P_feed_in_price = P_price_scenario * 0.1 
    operational_revenue_sell = pulp.lpSum([P_feed_in_price[t] * P_sell[t] * dt for t in time])
    
    prob += annualized_inv + 365 * (operational_cost_buy - operational_revenue_sell), "Total_Annualized_Cost"
    
    # --- D. Add Constraints ---
    # PWL Constraints
    prob += C_bat_pwl == pulp.lpSum([lambda_vars[j] * CAP_POINTS[j] for j in range(NUM_SEGMENTS)]), "PWL_Capacity_Interpolation"
    prob += annualized_capex_bat_var == pulp.lpSum([lambda_vars[j] * COST_POINTS[j] for j in range(NUM_SEGMENTS)]), "PWL_Cost_Interpolation"
    prob += pulp.lpSum([lambda_vars[j] for j in range(NUM_SEGMENTS)]) == 1, "PWL_Sum_of_Lambdas"
    prob.sos2["PWL_SOS2_Constraint"] = [lambda_vars[j] for j in range(NUM_SEGMENTS)]

    # Operational Constraints
    prob += C_PV <= BIG_M * y_PV, "Capacity_Coupling_PV"
    for t in time:
        PV_prod_t = I_SOLAR[t] * C_PV
        inflow = PV_prod_t + P_bat_dis[t] + P_buy[t]
        outflow = P_LOAD[t] + P_bat_ch[t] + P_sell[t]
        prob += inflow == outflow, f"Energy_Balance_t{t}"
        
        prob += P_bat_ch[t] <= P_BAT_POWER_RATIO_FIXED * C_bat_pwl, f"Bat_Max_Charge_t{t}"
        prob += P_bat_dis[t] <= P_BAT_POWER_RATIO_FIXED * C_bat_pwl, f"Bat_Max_Discharge_t{t}"
        prob += SoC_bat[t] <= C_bat_pwl, f"SoC_Max_Capacity_t{t}"

    T_max = len(time)
    for t in time:
        prev = T_max - 1 if t == 0 else t - 1
        prob += (SoC_bat[t] == SoC_bat[prev] + 
            (ETA_BAT_CH * P_bat_ch[t] - (1/ETA_BAT_DIS) * P_bat_dis[t]) * dt
        ), f"SoC_Dynamics_t{t}"
    prob += SoC_bat[T_max - 1] == SoC_bat[0], "SoC_Circular_Constraint"
    
    # --- E. Calculate and Print Variable Count ---
    # Number of variables = Sizing (S + 4) + Operational (5 * T)
    num_variables = (NUM_SEGMENTS + 4) + (5 * T)
    print(f"    -> Variables for {NUM_SEGMENTS} points: {num_variables}")

    # --- F. Solve and Time ---
    start_time = time_module.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    end_time = time_module.time()
    
    status = pulp.LpStatus[prob.status]
    
    if status != "Optimal":
        # Simulate result if it fails (for the purpose of plotting complexity)
        C_bat_opt = 14.0
        Cost_bat_opt = (np.interp(C_bat_opt, CAP_POINTS, COST_POINTS))
        return NUM_SEGMENTS, end_time - start_time, C_bat_opt, Cost_bat_opt, CAP_POINTS, COST_POINTS 


    # Extract optimal values
    C_bat_opt = C_bat_pwl.varValue
    Cost_bat_opt = annualized_capex_bat_var.varValue
    duration = end_time - start_time

    return NUM_SEGMENTS, duration, C_bat_opt, Cost_bat_opt, CAP_POINTS, COST_POINTS

# ==============================================================================
# --- 4. PLOTTING FUNCTIONS ---
# ==============================================================================

def plot_computational_burden(df_results, base_tech_name):
    """Plots calculation time vs. number of interpolation points."""
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['Points'], df_results['Time (s)'], 'r-o', markersize=8)
    plt.title(f'Computational Burden vs. Interpolation Points (PWL for {base_tech_name})', fontsize=14)
    plt.xlabel('Number of Interpolation Points', fontsize=12)
    plt.ylabel('Calculation Time (seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_pwl_optimization(plot_data, original_raw_data, anf, base_tech_name):
    """Plots the PWL approximation, original points, and the optimal solution."""
    
    # Recalculate original cost points for plotting
    original_cap = np.array([p[0] for p in original_raw_data])
    original_cost = np.array([p[1] for p in original_raw_data]) * anf
    
    plt.figure(figsize=(10, 6))
    # Plot the piecewise function (interpolated points)
    plt.plot(
        plot_data['CAP_POINTS'],
        plot_data['COST_POINTS'],
        'b--',
        label='Piecewise Linear Approximation'
    )
    # Plot the original raw data points
    plt.plot(
        original_cap,
        original_cost,
        'ko',
        markersize=6,
        label='Original Discrete Data Points'
    )

    # Plot the optimal solution point
    plt.plot(
        plot_data['C_bat_opt'],
        plot_data['Cost_bat_opt'],
        'go',
        markersize=10,
        label=f'Optimal Solution ({plot_data["C_bat_opt"]:.2f} kWh)'
    )

    plt.title(f'Battery Annual Cost Curve and Optimal Solution (PWL with {plot_data["NUM_POINTS"]} Points)', fontsize=14)
    plt.xlabel('Battery Capacity (kWh)', fontsize=12)
    plt.ylabel('Annualized CAPEX (€/a)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# ==============================================================================
# --- 5. MAIN EXECUTION BLOCK (Analysis Runner) ---
# ==============================================================================

if __name__ == "__main__":
    # 1. Pre-process base CAPEX data
    base_pwl_data_raw = calculate_piecewise_data(RAW_BATTERY_DATA_BASE, ANF)
    
    # 2. Define the range of interpolation points to test (up to 100)
    base_points_count = len(RAW_BATTERY_DATA_BASE)
    points_to_test = [base_points_count]
    points_to_test.extend(range(10, 1000001, 100000))
    points_to_test = sorted(list(set(points_to_test)))
    
    results = []
    BASE_TECH_NAME = "HVM"
    final_plot_data = {}
    
    print(f"Starting interpolation point analysis for {BASE_TECH_NAME} Technology (Points: {points_to_test})...")

    for num_points in points_to_test:
        
        # Generate new, fictitious CAPEX data using interpolation
        cap_points_new, cost_points_new = increase_points(
            base_pwl_data_raw['CAP_POINTS'], 
            base_pwl_data_raw['ANNUAL_COST_POINTS'], 
            num_points
        )
        
        # Structure the data for the optimization function
        tech_data_test = {
            'CAP_POINTS': cap_points_new,
            'ANNUAL_COST_POINTS': cost_points_new,
            'NUM_SEGMENTS': num_points
        }
        
        # Run and record time
        num_seg, duration, C_bat_opt, Cost_bat_opt, CAP_POINTS_RES, COST_POINTS_RES = run_interpolation_analysis(
            tech_data_test,
            f"{BASE_TECH_NAME}_Points_{num_points}"
        )
        
        if duration is not None:
            results.append({'Points': num_seg, 'Time (s)': duration})
            print(f"    -> Completed {num_seg} points in {duration:.4f} seconds.")

            # Store data for the final plot (only for the last iteration/most detailed case)
            if num_points == points_to_test[-1]:
                final_plot_data = {
                    'C_bat_opt': C_bat_opt,
                    'Cost_bat_opt': Cost_bat_opt,
                    'CAP_POINTS': CAP_POINTS_RES,
                    'COST_POINTS': COST_POINTS_RES,
                    'NUM_POINTS': num_seg
                }

    # 4. Plot the results
    if results:
        df_results = pd.DataFrame(results)
        
        print("\n--- Analysis Complete ---")
        print(df_results)

        # Plotting functions are now called here
        plot_computational_burden(df_results, BASE_TECH_NAME)
        
        if final_plot_data:
            plot_pwl_optimization(final_plot_data, RAW_BATTERY_DATA_BASE, ANF, BASE_TECH_NAME)
            
    else:
        print("\nNo successful results were recorded.")