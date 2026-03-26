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

# Investment costs (annualized)
CAPEX_BAT_ANNUAL = 56.17 # €/kWh 
CAPEX_PV_ANNUAL = 80 #250 #78.05 # €/kW_p

# Battery parameters
ETA_BAT_CH = 0.95
ETA_BAT_DIS = 0.95
P_BAT_POWER_RATIO = 1 # kW per kWh (max power)

# Demand and Solar Profiles (kept consistent for all scenarios)

P_LOAD = np.array([0.5, 0.5, 0.5, 0.6, 0.8, 1.2, 2.5, 3.5, 4.5, 5.0, 5.5, 5.8, 6.0, 5.9, 5.5, 5.2, 4.8, 3.5, 1.5, 1.0, 0.7, 0.6, 0.5, 0.5])
P_LOAD[P_LOAD < 0] = 0
I_SOLAR = np.array([max(0, np.sin(2*np.pi*(t-6)/24)) for t in time])


# ==============================================================================
# --- 2. DATA PREPARATION (Functions for prices and scenarios) ---
# ==============================================================================

def load_price_data(filename="load_data.csv", time_horizon=T):
    """Loads, cleans, and processes the electricity price data from file.

    It now reverses the data to be in chronological order (0h to 23h).
    """
    
    # Placeholder/Fallback array
    P_price_base_fallback = np.array([0.15] * 8 + [0.25] * 8 + [0.20] * 8)
    
    try:
        # 1. Load the CSV file. Use 'windows-1252' for Euro symbol compatibility.
        try:
            # Try semicolon
            price_df = pd.read_csv(filename, sep=';', encoding='windows-1252')
        except:
            # Fall back to comma
            price_df = pd.read_csv(filename, sep=',', encoding='windows-1252')
            
        # Ensure only the first two columns are used and name them explicitly
        if price_df.shape[1] >= 2:
            price_df = price_df.iloc[:, 0:2]
            price_df.columns = ['Date', 'Euro']
        else:
            raise ValueError("CSV must contain at least two columns.")

        # 2. Clean and convert the 'Euro' column.
        price_df['Euro_Cleaned'] = (
            price_df['Euro']
            .astype(str)
            .str.replace(r'[^\d,]', '', regex=True) # Remove all non-digits/commas
            .str.replace(',', '.', regex=False)    # Change comma decimal to dot decimal
            .astype(float)
        )
        
        # 3. Convert from Euro cents (€c) to Euro per kWh (€/kWh)
        P_price_data = (price_df['Euro_Cleaned'].to_numpy() / 1000)

        # 4. CRITICAL STEP: Reverse the array so it starts at hour 0
        P_price_base = P_price_data[::-1]
        
        # 5. Check data length
        if len(P_price_base) < time_horizon:
            print(f"Warning: Price data contains only {len(P_price_base)} points. Padded to {time_horizon} hours with the last available price.")
            P_price_base = np.pad(
                P_price_base, 
                (0, time_horizon - len(P_price_base)), 
                mode='edge' # Use 'edge' mode to repeat the last value
            )
        elif len(P_price_base) > time_horizon:
             P_price_base = P_price_base[:time_horizon]
        
        return {
            "Base": P_price_base
        }
        
    except FileNotFoundError:
        print(f"Warning: Could not load '{filename}'. File not found. Using placeholder price (Base).")
        return {
            "Base": P_price_base_fallback
        }
    except Exception as e:
        print(f"Error processing '{filename}': {e}. Using placeholder price (Base).")
        return {
            "Base": P_price_base_fallback
        }

def define_scenarios(base_prices):
    """
    Defines multiple P_grid price scenarios for simulation.
    """
    P_price_base = base_prices['Base']
    
    scenarios = {
        "1. Base TOU Price (Actual Data)": P_price_base,
        "2. Flat Price": np.full(T, np.mean(P_price_base)),
        "3. High Peak Price (18h-21h)": np.where(
            (time >= 18) & (time <= 21),
            P_price_base * 2.0,
            P_price_base
        )
    }
    return scenarios


# ==============================================================================
# --- 3. CORE OPTIMIZATION & PLOTTING FUNCTION ---
# ==============================================================================

def run_scenario_optimization(P_price_scenario, scenario_name):
    """
    Builds, solves the MILP model for a given price scenario, and plots results.
    """
    
    # --- A. Initialize Problem ---
    prob = pulp.LpProblem(f"Energy_System_MILP_{scenario_name}", pulp.LpMinimize)
    
    # --- B. Define Variables ---
    # Design Variables (Capacities)
    C_PV = pulp.LpVariable('C_PV', lowBound=0)
    C_bat = pulp.LpVariable('C_bat', lowBound=0)
    # Binary Variables (Installation Choice)
    y_PV = pulp.LpVariable('y_PV', cat='Binary')
    y_bat = pulp.LpVariable('y_bat', cat='Binary')
    
    # Operational Variables (Flows)
    P_buy = pulp.LpVariable.dicts('P_buy', time, lowBound=0) 
    P_sell = pulp.LpVariable.dicts('P_sell', time, lowBound=0)
    
    P_bat_ch = pulp.LpVariable.dicts('P_bat_ch', time, lowBound=0)
    P_bat_dis = pulp.LpVariable.dicts('P_bat_dis', time, lowBound=0)
    SoC_bat = pulp.LpVariable.dicts('SoC_bat', time, lowBound=0)
    
    # --- C. Set Objective (Cost Minimization) ---
    
    # 1. Investment Costs (Annualized)
    annualized_inv = CAPEX_PV_ANNUAL * C_PV + CAPEX_BAT_ANNUAL * C_bat
    
    # 2. Operational Costs (Buying)
    operational_cost_buy = pulp.lpSum([P_price_scenario[t] * P_buy[t] * dt for t in time])
    
    # 3. Operational Revenue (Selling)
    P_feed_in_price = P_price_scenario * 0.1 
    operational_revenue_sell = pulp.lpSum([P_feed_in_price[t] * P_sell[t] * dt for t in time])
    
    # Total Objective: Minimize Inv + (Cost to Buy - Revenue from Sell) * 365
    prob += annualized_inv + 365 * (operational_cost_buy - operational_revenue_sell), "Total_Annualized_Cost"
    
    # --- D. Add Constraints ---
    # Capacity Coupling (Big M for linking capacity to installation choice)
    prob += C_PV <= BIG_M * y_PV, "Capacity_Coupling_PV"
    prob += C_bat <= BIG_M * y_bat, "Capacity_Coupling_Bat"

    # Energy Balance (Power In = Power Out)
    for t in time:
        PV_prod_t = I_SOLAR[t] * C_PV
        
        # Inflow: PV + Battery Discharge + Grid Buy
        
        inflow = PV_prod_t + P_bat_dis[t] + P_buy[t]
        
        # Outflow: Load + Battery Charge + Grid Sell
        outflow = P_LOAD[t] + P_bat_ch[t] + P_sell[t]
        
        prob += inflow == outflow, f"Energy_Balance_t{t}"
        
        # Battery Limits
        prob += P_bat_ch[t] <= P_BAT_POWER_RATIO * C_bat, f"Bat_Max_Charge_t{t}"
        prob += P_bat_dis[t] <= P_BAT_POWER_RATIO * C_bat, f"Bat_Max_Discharge_t{t}"
        prob += SoC_bat[t] <= C_bat, f"SoC_Max_Capacity_t{t}"

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
        C_bat_opt = pulp.value(C_bat)
        results = {
            'C_PV': C_PV_opt,
            'C_bat': C_bat_opt,
            'Objective': pulp.value(prob.objective),
            'SoC_bat': np.array([pulp.value(SoC_bat[t]) for t in time]),
            'P_buy': np.array([pulp.value(P_buy[t]) for t in time]),
            'P_sell': np.array([pulp.value(P_sell[t]) for t in time]),
            'P_bat_ch': np.array([pulp.value(P_bat_ch[t]) for t in time]),
            'P_bat_dis': np.array([pulp.value(P_bat_dis[t]) for t in time]),
            'PV_prod': I_SOLAR * C_PV_opt,
            'P_load': P_LOAD
        }
    else:
        print(f"ERROR: Scenario '{scenario_name}' failed to reach optimal status: {status}")
        return

    # --- G. Plotting (Optimized for Slides) ---
    C_PV_opt = results['C_PV']
    C_bat_opt = results['C_bat']
    
    # Increased figure size for better resolution
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    # Main Title - Larger for slides
    plt.suptitle(f"Scenario: {scenario_name}\nCapacity: PV {C_PV_opt:.1f} kWp | Bat {C_bat_opt:.1f} kWh", 
                 fontsize=20, fontweight='bold', y=0.98)

    # Subplot 1: Power Flows
    axes[0].plot(time, results['P_load'], label="Load", color='black', linestyle='--', linewidth=3)
    axes[0].plot(time, results['PV_prod'], label="PV Gen", color='orange', linewidth=4)
    
    P_grid_net = results['P_buy'] - results['P_sell']
    axes[0].plot(time, P_grid_net, label="Grid Net", color='red', linewidth=3)
    
    P_bat_net = results['P_bat_dis'] - results['P_bat_ch']
    axes[0].plot(time, P_bat_net, label="Bat Net", color='green', linewidth=3)
    
    axes[0].set_ylabel("Power [kW]", fontsize=16)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    # LEGEND: Large font, horizontal layout to save space
    axes[0].legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.25), 
                   ncol=4, frameon=True, shadow=True)

    # Subplot 2: Battery State (kWh)
    axes[1].plot(time, results['SoC_bat'], label="Battery SoC", color='blue', linewidth=4)
    axes[1].axhline(y=C_bat_opt, color='blue', linestyle='--', alpha=0.5, label="Max Capacity")
    axes[1].set_ylabel("Energy [kWh]", fontsize=16)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=16, loc='upper right', frameon=True, shadow=True)
    
    # Subplot 3: Price Profile
    axes[2].plot(time, P_price_scenario, label="Buy Price", color='purple', linewidth=4)
    axes[2].plot(time, P_feed_in_price, label="Sell Price", color='gray', linestyle='--', linewidth=2)
    axes[2].set_ylabel("Price [€/kWh]", fontsize=16)
    axes[2].set_xlabel("Time [h]", fontsize=16)
    axes[2].tick_params(axis='both', which='major', labelsize=14)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(fontsize=16, loc='upper right', frameon=True, shadow=True)

    # Adjust layout to make room for the top legend
    plt.tight_layout(rect=[0, 0, 1, 0.93]) 
    plt.show()

# ==============================================================================
# --- 4. MAIN EXECUTION BLOCK (Scenario Runner) ---
# ==============================================================================

if __name__ == "__main__":
    # 1. Load data and define scenarios
    base_prices = load_price_data()
    all_scenarios = define_scenarios(base_prices)

    # 2. Iterate through all defined scenarios
    print("Starting scenario simulation...")
    for name, price_array in all_scenarios.items():
        print(f"\n--- Running Scenario: {name} ---")
        run_scenario_optimization(price_array, name)
        
    print("\nScenario simulation complete.")