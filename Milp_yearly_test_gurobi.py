import numpy as np
import pandas as pd
import math
import requests
import json
import gurobipy as gp  # 2. Now it will find your unrestricted academic license
from gurobipy import GRB
from Fetch_and_save_data import (load_pvgis_from_csv, load_prices_from_csv,
                                  load_prices_from_epex, fetch_pvgis_to_csv)

# ==============================================================================
# --- 1. GLOBAL PARAMETERS & PHYSICAL CONSTANTS ---
# ==============================================================================

YEAR        = 2023
T           = 8760        # hours in a year

# --- Price source selection ---
# "csv"  : load from prices_data.csv (fetched via Entsoe API, any year)
# "epex" : load from epex_2025.csv   (downloaded EPEX Belgium day-ahead 2025)
PRICE_SOURCE   = "csv"
EPEX_CSV_PATH  = "epex_2025.csv"
dt          = 1.0         # hours
BIG_M       = 1e6
MONTHS      = 12
VAT         = 0.21
Factor_Thermal = 0.46

# --- Electrical System ---
CAPEX_BAT_ANNUAL    = 800/20   # €/kWh/yr (Base rate, anchored at 10 kWh)
CAPEX_PV_ANNUAL     = 900/20   # €/kWp/yr (Base rate, anchored at 10 kWp)
C_CAP               = 5        # €/kW/month
ETA_BAT_CH          = 0.95
ETA_BAT_DIS         = 0.95
P_BAT_POWER_RATIO   = 0.5      # kW per kWh
TES_POWER_RATIO     = 0.5      # kW_th per kWh_th

# --- Design variable upper bounds ---
C_PV_MAX    = 50     # kWp
C_BAT_MAX   = 200    # kWh
C_HP_MAX    = 200     # kW_th
MAX_ANNUAL_CAPEX = 25000       # €/yr — Maximum allowed annualized CAPEX

# --- Heat Pump ---
CAPEX_HP_ANNUAL     = 350.0    # €/kW_th/yr (Base rate, anchored at 30 kW_th)
fixed_cost_annual   = 200      
HP_MIN_FRAC         = 0.30     

# --- Part-Load & Weather Parameters ---
USE_PARTLOAD        = False     
C_D                 = 0.25     

# --- PWL Segments (PLR vs EIR) ---
PWL_PLR_BOUNDS = [0.30, 0.60, 1.00] 
PWL_EIR_POINTS = [0.35, 0.60, 1.00]
NUM_PWL_SEGS   = len(PWL_PLR_BOUNDS) - 1

HP_MIN_FRAC    = PWL_PLR_BOUNDS[0]  

temp_h, temp_c, temp_env = 50, 30, 10   # °C — supply, return, ambient
eta_in,  eta_out         = 0.95, 0.95
dt_sec                   = 3600

# ==============================================================================
# --- 2a. LTES (LATENT THERMAL ENERGY STORAGE) PARAMETERS ---
# ==============================================================================
rho_pcm     = 860.0     # kg/m³
L_pcm       = 199.0     # kJ/kg
rho_L_pcm   = rho_pcm * L_pcm / 3600.0   # kWh/m³ ≈ 47.6 kWh/m³

CAPEX_TES_eu_kWh_LTES = 30.0   # €/kWh/yr (Base rate, anchored at 30 kWh_th)
LTES_LOSS_FRAC_HR     = 0.005  
C_LTES_MAX            = 1000 

# ==============================================================================
# --- 2b. WTES (SENSIBLE / STRATIFIED WATER TANK) PARAMETERS ---
# ==============================================================================
T_HIGH  = temp_h   
T_LOW   = temp_c   

delta_T_HC  = T_HIGH  - T_LOW          
delta_T_C0  = T_LOW   - temp_env       
delta_T_H0  = T_HIGH  - temp_env       

rho_water   = 971.8    # kg/m³
c_water     = 4190.0   # J/(kg·K)

d_wtes      = 1.0                                        
A_cross_wtes = math.pi * (d_wtes / 2) ** 2               

U_wall_wtes = 1.0    

kWh_per_m_wtes = (A_cross_wtes * rho_water * c_water * delta_T_HC) / 3.6e6

beta_wtes  = U_wall_wtes * (4.0 / (d_wtes * rho_water * c_water)) * dt_sec / 3.6e6
gamma_wtes = beta_wtes * (delta_T_C0 / delta_T_HC)
loss_lids_wtes = (
    U_wall_wtes * 2.0 * A_cross_wtes
    * ((delta_T_H0 + delta_T_C0) / 2.0)
    * dt_sec / 3.6e6
)

# Reference: 3 €/L at 500 L tank (literature).
# 500 L = 0.5 m³  →  one-time CAPEX = 3*1000/kWh_per_m3 = 66.3 €/kWh_th
# Annualised over 20 yr (same convention as BAT/PV): 66.3/20 ≈ 3.3 €/kWh/yr
WTES_REF_LITRES         = 500.0                                     # L — anchor tank size
WTES_REF_COST_EUR_L     = 3.0                                       # €/L one-time
CAPEX_TES_eu_kWh_WTES   = (WTES_REF_COST_EUR_L * 1000             # €/m³
                            / ((rho_water * c_water * delta_T_HC) / 3.6e6)  # kWh/m³
                            ) / 20                                  # annualised €/kWh/yr
CAPEX_WTES_METER_ANNUAL = CAPEX_TES_eu_kWh_WTES * kWh_per_m_wtes  # €/m/yr (kept for reference)

h_wtes_max = 20.0    # m

# ==============================================================================
# --- PWL CAPEX PARAMETER GENERATOR ---
# ==============================================================================
NUM_CAPEX_SEGS = 5  # You can adjust this to 4, 5, 6, etc.

def generate_capex_pwl(max_cap, base_rate, anchor_cap, is_wtes=False):
    """
    Generates EOS breakpoints using a power-law curve (scaling factor = 0.6).

    Parameters
    ----------
    max_cap    : upper bound of the decision variable (kWh, kWp, kW_th, or m for WTES)
    base_rate  : annualised unit cost AT the anchor [€/unit/yr]
    anchor_cap : reference capacity where base_rate is exactly true [same unit as max_cap,
                 or litres for WTES — converted internally]
    is_wtes    : if True, x-axis is in metres but cost curve is built in kWh_th
    """
    scaling_factor = 0.6  # industry-standard power-law exponent

    # For WTES: work in kWh_th internally, convert anchor from litres → kWh_th
    if is_wtes:
        max_cap_kwh  = float(max_cap)  * kWh_per_m_wtes        # m → kWh_th
        anchor_kwh   = (anchor_cap / 1000.0) * ((rho_water * c_water * delta_T_HC) / 3.6e6)
    else:
        max_cap_kwh  = float(max_cap)
        anchor_kwh   = float(anchor_cap)

    # Quadratic spacing clusters breakpoints near zero where the curve bends most
    fractions = np.linspace(0, 1, NUM_CAPEX_SEGS) ** 2
    bp_x_kwh  = (fractions * max_cap_kwh).tolist()

    anchor_total_cost = anchor_kwh * base_rate   # € /yr at anchor

    bp_y = [0.0 if x == 0 else anchor_total_cost * (x / anchor_kwh) ** scaling_factor
            for x in bp_x_kwh]

    # Convert x back to metres for the WTES solver variable
    bp_x = [x / kWh_per_m_wtes for x in bp_x_kwh] if is_wtes else bp_x_kwh

    return bp_x, bp_y

BP_X_PV,   BP_Y_PV   = generate_capex_pwl(C_PV_MAX,    CAPEX_PV_ANNUAL,          anchor_cap=10.0)
BP_X_BAT,  BP_Y_BAT  = generate_capex_pwl(C_BAT_MAX,   CAPEX_BAT_ANNUAL,         anchor_cap=10.0)
BP_X_HP,   BP_Y_HP   = generate_capex_pwl(C_HP_MAX,    CAPEX_HP_ANNUAL,          anchor_cap=30.0)
BP_X_LTES, BP_Y_LTES = generate_capex_pwl(C_LTES_MAX,  CAPEX_TES_eu_kWh_LTES,   anchor_cap=30.0)
# WTES anchor_cap is in litres — converted to kWh_th internally
BP_X_WTES, BP_Y_WTES = generate_capex_pwl(h_wtes_max,  CAPEX_TES_eu_kWh_WTES,   anchor_cap=WTES_REF_LITRES, is_wtes=True)

# ==============================================================================
# --- 3. DEMAND PROFILES ---
# ==============================================================================
GAS_CSV_PATH    = r'C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\gasdata.csv'
GAS_DEMAND_YEAR = 2021

def load_demand_from_gascsv(csv_path=GAS_CSV_PATH, demand_year=GAS_DEMAND_YEAR):
    df_csv = pd.read_csv(
        csv_path, sep=';', skiprows=1, header=None,
        names=['timestamp', 'col2', 'demand_kW_raw', 'extra'],
        decimal=',', encoding='utf-8'
    )
    df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'], utc=True)
    df_csv['timestamp'] = (df_csv['timestamp']
                           .dt.tz_convert('Europe/Brussels')
                           .dt.tz_localize(None))
    df_csv = df_csv.set_index('timestamp')
    df_csv['demand_kW'] = pd.to_numeric(df_csv['demand_kW_raw'], errors='coerce')

    mask = df_csv.index.year == demand_year
    sub  = df_csv.loc[mask, 'demand_kW'].resample('h').mean()

    print(f"Gas CSV loaded: {len(sub)} hourly records for {demand_year}")
    target_index = pd.date_range(f"{demand_year}-01-01", periods=T, freq='h')
    sub = sub.reindex(target_index)
    if sub.isna().sum() > 0:
        sub = sub.ffill().bfill()

    thermal    = sub.to_numpy(dtype=float)
    electrical = 90/260 * thermal
    return Factor_Thermal * thermal, electrical

P_THERMAL_LOAD, P_LOAD = load_demand_from_gascsv(GAS_CSV_PATH, GAS_DEMAND_YEAR)

dates_hourly  = pd.date_range(f"{YEAR}-01-01", periods=T, freq='h')
month_of_hour = dates_hourly.month.values - 1   

# ==============================================================================
# --- 4. HELPER FUNCTIONS ---
# ==============================================================================
def calculate_capacity_fraction(T_amb_array):
    T_ref = 7.0   
    k = 0.0146    
    Cap_frac_t = 1.0 + k * (T_amb_array - T_ref)
    return np.clip(Cap_frac_t, a_min=0.678, a_max=1.15)

# ==============================================================================
# --- 5. INTEGRATED MILP (GUROBIPY) ---
# ==============================================================================
def run_integrated_optimization(I_SOLAR, COP_t, Cap_frac_t, P_price_buy, P_price_sell):
    # Initialize native Gurobi Model
    model = gp.Model("Integrated_MILP_Annual_TES_Choice")

    # ------------------------------------------------------------------
    # Design variables (Scalars)
    # ------------------------------------------------------------------
    C_PV   = model.addVar(lb=0, ub=C_PV_MAX, name="C_PV")
    C_bat  = model.addVar(lb=0, ub=C_BAT_MAX, name="C_bat")
    C_HP   = model.addVar(lb=0, ub=C_HP_MAX, name="C_HP")
    C_ltes = model.addVar(lb=0, ub=C_LTES_MAX, name="C_ltes")
    h_wtes = model.addVar(lb=0, ub=h_wtes_max, name="h_wtes")

    y_PV   = model.addVar(vtype=GRB.BINARY, name="y_PV")
    y_bat  = model.addVar(vtype=GRB.BINARY, name="y_bat")
    y_HP   = model.addVar(vtype=GRB.BINARY, name="y_HP")
    y_ltes = model.addVar(vtype=GRB.BINARY, name="y_ltes")
    y_wtes = model.addVar(vtype=GRB.BINARY, name="y_wtes")

    P_peak_m = model.addVars(MONTHS, lb=0, name="P_peak_m")

    # Total cost variables for the objective
    cost_pv   = model.addVar(lb=0, name="cost_pv")
    cost_bat  = model.addVar(lb=0, name="cost_bat")
    cost_hp   = model.addVar(lb=0, name="cost_hp")
    cost_ltes = model.addVar(lb=0, name="cost_ltes")
    cost_wtes = model.addVar(lb=0, name="cost_wtes")

    # ------------------------------------------------------------------
    # Operational variables (Hourly vectors)
    # ------------------------------------------------------------------
    P_buy      = model.addVars(T, lb=0, name="P_buy")
    P_sell     = model.addVars(T, lb=0, name="P_sell")
    P_bat_ch   = model.addVars(T, lb=0, name="P_bat_ch")
    P_bat_dis  = model.addVars(T, lb=0, name="P_bat_dis")
    SoC_bat    = model.addVars(T, lb=0, name="SoC_bat")
    P_hp_elec  = model.addVars(T, lb=0, name="P_hp_elec")
    
    q_hp_th      = model.addVars(T, lb=0, name="q_hp_th")
    C_hp_active  = model.addVars(T, lb=0, name="C_hp_active")
    
    Q_load_in    = model.addVars(T, lb=0, name="Q_load_in")
    Q_ltes_in    = model.addVars(T, lb=0, name="Q_ltes_in")
    Q_ltes_out   = model.addVars(T, lb=0, name="Q_ltes_out")
    Q_wtes_in    = model.addVars(T, lb=0, name="Q_wtes_in")
    Q_wtes_out   = model.addVars(T, lb=0, name="Q_wtes_out")
    
    if USE_PARTLOAD:
        u_hp_seg = model.addVars(T, NUM_PWL_SEGS, vtype=GRB.BINARY, name="u_hp_seg")
        q_hp_seg = model.addVars(T, NUM_PWL_SEGS, lb=0, name="q_hp_seg")
        c_hp_seg = model.addVars(T, NUM_PWL_SEGS, lb=0, name="c_hp_seg")

    Q_ltes     = model.addVars(T, lb=0, name="Q_ltes")
    Q_wtes     = model.addVars(T, lb=0, name="Q_wtes")

    u_hp       = model.addVars(T, vtype=GRB.BINARY, name="u_hp")
    u_bat_ch   = model.addVars(T, vtype=GRB.BINARY, name="u_bat_ch")
    u_bat_dis  = model.addVars(T, vtype=GRB.BINARY, name="u_bat_dis")
    u_ltes_in  = model.addVars(T, vtype=GRB.BINARY, name="u_ltes_in")
    u_ltes_out = model.addVars(T, vtype=GRB.BINARY, name="u_ltes_out")
    u_wtes_in  = model.addVars(T, vtype=GRB.BINARY, name="u_wtes_in")
    u_wtes_out = model.addVars(T, vtype=GRB.BINARY, name="u_wtes_out")

    # ------------------------------------------------------------------
    # Objective: minimise total annualised cost
    # ------------------------------------------------------------------
    C_WTES = kWh_per_m_wtes * h_wtes   

    inv_cost   = cost_pv + cost_bat + cost_hp + cost_ltes + cost_wtes
    
    # --- MAX ANNUAL CAPEX CONSTRAINT ---
    model.addConstr(inv_cost <= MAX_ANNUAL_CAPEX, "Max_Annualized_CAPEX")

    op_buy     = gp.quicksum(P_price_buy[t]  * P_buy[t]  * dt for t in range(T))
    op_sell    = gp.quicksum(P_price_sell[t] * P_sell[t] * dt for t in range(T))
    cap_tariff = gp.quicksum(C_CAP * P_peak_m[m] for m in range(MONTHS))

    model.setObjective(
        (inv_cost + (op_buy - op_sell + cap_tariff + fixed_cost_annual) * (1 + VAT)), 
        GRB.MINIMIZE
    )

    # ------------------------------------------------------------------
    # Technology selection & PWL CAPEX Constraints
    # ------------------------------------------------------------------
    model.addConstr(C_PV  <= C_PV_MAX  * y_PV,  "Cap_PV")
    model.addConstr(C_bat <= C_BAT_MAX * y_bat, "Cap_bat")
    model.addConstr(C_HP  <= C_HP_MAX  * y_HP,  "Cap_HP")
    model.addConstr(y_ltes + y_wtes == 1, "TES_type_select_one")
    model.addConstr(C_ltes <= C_LTES_MAX * y_ltes, "LTES_select")
    model.addConstr(h_wtes <= h_wtes_max * y_wtes, "WTES_select")

    # Native Gurobi Piecewise-Linear constraints for Capex
    model.addGenConstrPWL(C_PV,   cost_pv,   BP_X_PV,   BP_Y_PV,   "PWL_PV")
    model.addGenConstrPWL(C_bat,  cost_bat,  BP_X_BAT,  BP_Y_BAT,  "PWL_BAT")
    model.addGenConstrPWL(C_HP,   cost_hp,   BP_X_HP,   BP_Y_HP,   "PWL_HP")
    model.addGenConstrPWL(C_ltes, cost_ltes, BP_X_LTES, BP_Y_LTES, "PWL_LTES")
    model.addGenConstrPWL(h_wtes, cost_wtes, BP_X_WTES, BP_Y_WTES, "PWL_WTES")

    # ------------------------------------------------------------------
    # Hourly constraints
    # ------------------------------------------------------------------
    print("Building hourly constraints (8 760 steps) …")
    for t in range(T):
        prev = T - 1 if t == 0 else t - 1
        m    = int(month_of_hour[t])

        PV_t = I_SOLAR[t] * C_PV
     
        # --- Heat Pump Operation Logic ---
        if USE_PARTLOAD:
            model.addConstr(C_hp_active[t] <= C_HP_MAX * u_hp[t], f"HPAct_Zero_{t}")
            model.addConstr(C_hp_active[t] <= C_HP, f"HPAct_Cap_{t}")
            model.addConstr(C_hp_active[t] >= C_HP - C_HP_MAX * (1 - u_hp[t]), f"HPAct_Match_{t}")

            model.addConstr(gp.quicksum(u_hp_seg[t, s] for s in range(NUM_PWL_SEGS)) == u_hp[t], f"HPSeg_Active_{t}")
            model.addConstr(gp.quicksum(c_hp_seg[t, s] for s in range(NUM_PWL_SEGS)) == C_hp_active[t], f"HPSeg_Cap_Sum_{t}")

            p_elec_total_expr = 0
            q_th_total_expr = 0

            for s in range(NUM_PWL_SEGS):
                model.addConstr(c_hp_seg[t, s] <= C_HP_MAX * u_hp_seg[t, s], f"HPSeg_Cap_Max_{t}_{s}")
                
                plr_min, plr_max = PWL_PLR_BOUNDS[s], PWL_PLR_BOUNDS[s+1]
                eir_min, eir_max = PWL_EIR_POINTS[s], PWL_EIR_POINTS[s+1]
                
                slope = (eir_max - eir_min) / (plr_max - plr_min)
                intercept = eir_min - slope * plr_min
                cap_t = c_hp_seg[t, s] * Cap_frac_t[t]
                
                model.addConstr(q_hp_seg[t, s] >= plr_min * cap_t, f"HPSeg_QMin_{t}_{s}")
                model.addConstr(q_hp_seg[t, s] <= plr_max * cap_t, f"HPSeg_QMax_{t}_{s}")
                
                q_th_total_expr += q_hp_seg[t, s]
                p_elec_total_expr += (slope * q_hp_seg[t, s]) + (intercept * cap_t)

            model.addConstr(q_hp_th[t] == q_th_total_expr, f"HP_Q_Total_{t}")
            model.addConstr(P_hp_elec[t] * COP_t[t] == p_elec_total_expr, f"HP_P_Total_{t}")

        else:
            model.addConstr(C_hp_active[t] <= C_HP_MAX * u_hp[t], f"HPAct_Zero_{t}")
            model.addConstr(C_hp_active[t] <= C_HP, f"HPAct_Cap_{t}")
            model.addConstr(C_hp_active[t] >= C_HP - C_HP_MAX * (1 - u_hp[t]), f"HPAct_Match_{t}")

            model.addConstr(q_hp_th[t] <= C_hp_active[t] * Cap_frac_t[t], f"HPMax_Weather_{t}")
            model.addConstr(q_hp_th[t] >= HP_MIN_FRAC * C_hp_active[t] * Cap_frac_t[t], f"HPMin_Weather_{t}")
            model.addConstr(P_hp_elec[t] * COP_t[t] == (C_D * C_hp_active[t] * Cap_frac_t[t]) + ((1 - C_D) * q_hp_th[t]), f"HPPenalty_{t}")

        # --- Electrical balance ---
        model.addConstr(PV_t + P_bat_dis[t] + P_buy[t] == P_LOAD[t] + P_bat_ch[t] + P_hp_elec[t] + P_sell[t], f"Elec_{t}")
        model.addConstr(P_sell[t] <= PV_t + P_bat_dis[t], f"SellMax_{t}")

        # --- Battery dynamics ---
        model.addConstr(u_bat_ch[t] + u_bat_dis[t] <= 1, f"BatExcl_{t}")
        model.addConstr(P_bat_ch[t]  <= P_BAT_POWER_RATIO * C_BAT_MAX * u_bat_ch[t], f"BatChMax_{t}")
        model.addConstr(P_bat_dis[t] <= P_BAT_POWER_RATIO * C_BAT_MAX * u_bat_dis[t], f"BatDisMax_{t}")
        model.addConstr(P_bat_ch[t]  <= P_BAT_POWER_RATIO * C_bat, f"BatChCap_{t}")
        model.addConstr(P_bat_dis[t] <= P_BAT_POWER_RATIO * C_bat, f"BatDisCap_{t}")
        model.addConstr(SoC_bat[t]   <= C_bat, f"SoCMax_{t}")
        model.addConstr(SoC_bat[t]   == SoC_bat[prev] + (ETA_BAT_CH * P_bat_ch[t] - (1.0 / ETA_BAT_DIS) * P_bat_dis[t]) * dt, f"SoCDyn_{t}")

        # ---- LTES dynamics ----
        loss_ltes_t = LTES_LOSS_FRAC_HR * C_ltes

        model.addConstr(Q_ltes[t] == Q_ltes[prev] - loss_ltes_t + Q_ltes_in[t] - Q_ltes_out[t], f"LTESDyn_{t}")
        model.addConstr(u_ltes_in[t] + u_ltes_out[t] <= y_ltes, f"LTESExcl_{t}")
        model.addConstr(Q_ltes_in[t]  <= BIG_M * u_ltes_in[t], f"LTESInBin_{t}")
        model.addConstr(Q_ltes_out[t] <= BIG_M * u_ltes_out[t], f"LTESOutBin_{t}")
        model.addConstr(Q_ltes_in[t]  <= TES_POWER_RATIO * C_ltes, f"LTESInCap_{t}")
        model.addConstr(Q_ltes_out[t] <= TES_POWER_RATIO * C_ltes, f"LTESOutCap_{t}")
        model.addConstr(Q_ltes[t] >= 0.05 * C_ltes - BIG_M * (1 - y_ltes), f"LTESMin_{t}")
        model.addConstr(Q_ltes[t] <= 0.95 * C_ltes + BIG_M * (1 - y_ltes), f"LTESMax_{t}")
        model.addConstr(Q_ltes[t] <= BIG_M * y_ltes, f"LTESZero_{t}")

        # ---- WTES dynamics ----
        loss_wtes_t = (beta_wtes * Q_wtes[prev] + gamma_wtes * C_WTES + loss_lids_wtes * y_wtes)

        model.addConstr(Q_wtes[t] == Q_wtes[prev] - loss_wtes_t + Q_wtes_in[t] - Q_wtes_out[t], f"WTESDyn_{t}")
        model.addConstr(u_wtes_in[t] + u_wtes_out[t] <= y_wtes, f"WTESExcl_{t}")
        model.addConstr(Q_wtes_in[t]  <= BIG_M * u_wtes_in[t], f"WTESInBin_{t}")
        model.addConstr(Q_wtes_out[t] <= BIG_M * u_wtes_out[t], f"WTESOutBin_{t}")
        model.addConstr(Q_wtes_in[t]  <= TES_POWER_RATIO * C_WTES, f"WTESInCap_{t}")
        model.addConstr(Q_wtes_out[t] <= TES_POWER_RATIO * C_WTES, f"WTESOutCap_{t}")
        model.addConstr(Q_wtes[t] >= 0.05 * C_WTES - BIG_M * (1 - y_wtes), f"WTESMin_{t}")
        model.addConstr(Q_wtes[t] <= 0.95 * C_WTES + BIG_M * (1 - y_wtes), f"WTESMax_{t}")
        model.addConstr(Q_wtes[t] <= BIG_M * y_wtes, f"WTESZero_{t}")

        # ---- Thermal balance ----
        model.addConstr(q_hp_th[t] == Q_ltes_in[t] + Q_wtes_in[t] + Q_load_in[t], f"HPSplit_{t}")
        model.addConstr(Q_load_in[t] + Q_ltes_out[t] + Q_wtes_out[t] == P_THERMAL_LOAD[t], f"ThermLoad_{t}")

        # --- Monthly peak tracking ---
        model.addConstr(P_peak_m[m] >= P_buy[t], f"PeakM_{t}")

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    print("\nSolving (this may take several minutes for 8 760-step MILP) …")
    model.setParam('MIPGap', 0.01)
    model.setParam('TimeLimit', 300)
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        print("\nSolver failed to find an optimal/feasible solution.")
        return None, None

    # ------------------------------------------------------------------
    # Extract results
    # ------------------------------------------------------------------
    y_ltes_val = int(round(y_ltes.X))
    y_wtes_val = int(round(y_wtes.X))
    tes_type   = "LTES (PCM)" if y_ltes_val else "WTES (stratified water)"

    C_LTES_val   = C_ltes.X
    V_ltes_val   = C_LTES_val / rho_L_pcm if rho_L_pcm else 0.0
    mass_pcm     = rho_pcm * V_ltes_val

    h_wtes_val   = h_wtes.X
    C_WTES_val   = kWh_per_m_wtes * h_wtes_val
    V_wtes_val   = A_cross_wtes   * h_wtes_val
    mass_water   = rho_water      * V_wtes_val

    if y_ltes_val:
        C_TES_val = C_LTES_val
        V_tes_val = V_ltes_val
    else:
        C_TES_val = C_WTES_val
        V_tes_val = V_wtes_val

    opt = {
        'C_PV':       C_PV.X,
        'C_bat':      C_bat.X,
        'C_HP':       C_HP.X,
        'tes_type':   tes_type,
        'y_ltes':     y_ltes_val,
        'y_wtes':     y_wtes_val,
        'C_ltes':     C_LTES_val,
        'h_wtes':     h_wtes_val,
        'C_LTES':     C_LTES_val,
        'C_WTES':     C_WTES_val,
        'V_ltes':     V_ltes_val,
        'V_wtes':     V_wtes_val,
        'mass_pcm':   mass_pcm,
        'mass_water': mass_water,
        'C_TES':      C_TES_val,
        'V_tes':      V_tes_val,
        'P_peak_m':   [P_peak_m[m].X for m in range(MONTHS)],
        'obj':        model.ObjVal,
    }
    opt['P_peak_annual'] = max(opt['P_peak_m'])

    arr = lambda d: np.array([d[t].X for t in range(T)])
    Q_tes_active = Q_ltes if y_ltes_val else Q_wtes

    Q_tes_in_combined  = np.array([Q_ltes_in[t].X  + Q_wtes_in[t].X  for t in range(T)])
    Q_tes_out_combined = np.array([Q_ltes_out[t].X + Q_wtes_out[t].X for t in range(T)])

    res = {
        'SoC_bat':    arr(SoC_bat),
        'P_buy':      arr(P_buy),
        'P_sell':     arr(P_sell),
        'P_bat_ch':   arr(P_bat_ch),
        'P_bat_dis':  arr(P_bat_dis),
        'P_hp_elec':  arr(P_hp_elec),
        'Q_tes':      arr(Q_tes_active),
        'Q_ltes':     arr(Q_ltes),
        'Q_wtes':     arr(Q_wtes),
        'Q_ltes_in':  arr(Q_ltes_in),
        'Q_ltes_out': arr(Q_ltes_out),
        'Q_wtes_in':  arr(Q_wtes_in),
        'Q_wtes_out': arr(Q_wtes_out),
        'Q_tes_in':   Q_tes_in_combined,
        'Q_tes_out':  Q_tes_out_combined,
        'Q_load_in':  arr(Q_load_in),
        'PV_prod':    I_SOLAR * opt['C_PV'],
        'Q_hp_th':    arr(q_hp_th),              
        'Cap_frac_t': Cap_frac_t,                
    }

    if y_ltes_val:
        res['TES_loss'] = np.array([LTES_LOSS_FRAC_HR * C_LTES_val for t in range(T)])
    else:
        res['TES_loss'] = np.array([
            beta_wtes * res['Q_tes'][T-1 if t == 0 else t-1] + gamma_wtes * C_WTES_val + loss_lids_wtes
            for t in range(T)
        ])

    # Extract dynamic CAPEX directly from the variables
    capex_pv   = cost_pv.X
    capex_bat  = cost_bat.X
    capex_hp   = cost_hp.X
    capex_ltes = cost_ltes.X
    capex_wtes = cost_wtes.X
    capex_tes  = capex_ltes + capex_wtes   
    capex_total = capex_pv + capex_bat + capex_hp + capex_tes

    opex_buy      = sum(P_price_buy[t]  * res['P_buy'][t]  * dt for t in range(T))
    opex_sell     = sum(P_price_sell[t] * res['P_sell'][t] * dt for t in range(T))
    opex_cap_tar  = sum(C_CAP * pm for pm in opt['P_peak_m'])
    opex_fixed    = fixed_cost_annual
    opex_net      = (opex_buy - opex_sell + opex_cap_tar + opex_fixed) * (1 + VAT)

    C_WTES_max = kWh_per_m_wtes * h_wtes_max   # max WTES capacity at height limit

    print(f"\n=== OPTIMISATION RESULTS ===")
    print(f"  PV:             {opt['C_PV']:.2f} kWp")
    print(f"  Battery:        {opt['C_bat']:.2f} kWh")
    print(f"  HP:             {opt['C_HP']:.2f} kW_th")
    print(f"  TES selected:   {tes_type}  ({'✓ LTES' if y_ltes_val else '✓ WTES'} chosen by optimizer)")

    if y_ltes_val:
        at_limit = " ← AT MAX" if opt['C_ltes'] >= 0.99 * C_LTES_MAX else ""
        print(f"  LTES tank:      C_ltes  = {opt['C_ltes']:.2f} kWh_th  (max: {C_LTES_MAX:.0f} kWh_th){at_limit}")
        print(f"                  V_ltes  = {opt['V_ltes']:.3f} m³  ({opt['V_ltes']*1000:.0f} L)")
        print(f"                  m_PCM   = {opt['mass_pcm']:.0f} kg")
        print(f"  [not built]     WTES max capacity would have been: {C_WTES_max:.1f} kWh_th  (h={h_wtes_max:.0f} m, d={d_wtes:.1f} m)")
    else:
        at_limit = " ← AT HEIGHT LIMIT" if opt['h_wtes'] >= 0.99 * h_wtes_max else ""
        print(f"  WTES tank:      h_wtes  = {opt['h_wtes']:.3f} m  (max: {h_wtes_max:.0f} m){at_limit}")
        print(f"                  C_WTES  = {opt['C_WTES']:.2f} kWh_th  (max: {C_WTES_max:.1f} kWh_th)")
        print(f"                  V_wtes  = {opt['V_wtes']:.3f} m³  ({opt['V_wtes']*1000:.0f} L)")
        print(f"                  m_water = {opt['mass_water']:.0f} kg")
        print(f"  [not built]     LTES max capacity would have been: {C_LTES_MAX:.0f} kWh_th")

    print(f"\n--- CAPEX (annualised - PWL adjusted) ---")
    print(f"  PV:             €{capex_pv:>10,.0f}/yr")
    print(f"  Battery:        €{capex_bat:>10,.0f}/yr")
    print(f"  Heat Pump:      €{capex_hp:>10,.0f}/yr")
    if y_ltes_val:
        print(f"  LTES:           €{capex_ltes:>10,.0f}/yr")
    else:
        print(f"  WTES:           €{capex_wtes:>10,.0f}/yr")
    print(f"  ─────────────────────────")
    print(f"  Total CAPEX:    €{capex_total:>10,.0f}/yr")

    print(f"\n--- OPEX (annual, incl. {VAT*100:.0f}% VAT) ---")
    print(f"  Grid buy:               €{opex_buy*(1+VAT):>10,.0f}/yr  ({sum(res['P_buy']):.0f} kWh)")
    print(f"  Grid sell:              €{-opex_sell*(1+VAT):>10,.0f}/yr  ({sum(res['P_sell']):.0f} kWh)")
    print(f"  Capacity tariff:        €{opex_cap_tar*(1+VAT):>10,.0f}/yr")
    print(f"  Fixed connection cost:  €{opex_fixed*(1+VAT):>10,.0f}/yr")
    print(f"  ─────────────────────────")
    print(f"  Total OPEX (net):       €{opex_net:>10,.0f}/yr")

    print(f"\n--- TOTAL ---")
    print(f"  CAPEX + OPEX:           €{capex_total + opex_net:>10,.0f}/yr")
    print(f"  (Objective value):      €{opt['obj']:>10,.0f}/yr")

    print(f"\n--- ENERGY FLOWS ---")
    print(f"  Monthly peaks [kW]: {[f'{p:.1f}' for p in opt['P_peak_m']]}")
    print(f"  Annual grid buy:    {sum(res['P_buy']):.0f} kWh")
    print(f"  Annual grid sell:   {sum(res['P_sell']):.0f} kWh")
    print(f"  Annual PV gen:      {sum(res['PV_prod']):.0f} kWh")
    print(f"  Annual TES loss:    {sum(res['TES_loss']):.1f} kWh_th")
    print(f"  Annual Q_tes_in:    {sum(res['Q_tes_in']):.0f} kWh_th")
    print(f"  Annual Q_tes_out:   {sum(res['Q_tes_out']):.0f} kWh_th")
    print(f"  Annual Q_load_in:   {sum(res['Q_load_in']):.0f} kWh_th")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    np.savez('milp_results.npz',
             SoC_bat        = res['SoC_bat'],
             P_buy          = res['P_buy'],
             P_sell         = res['P_sell'],
             P_bat_ch       = res['P_bat_ch'],
             P_bat_dis      = res['P_bat_dis'],
             P_hp_elec      = res['P_hp_elec'],
             Q_tes          = res['Q_tes'],
             Q_ltes         = res['Q_ltes'],
             Q_wtes         = res['Q_wtes'],
             Q_ltes_in      = res['Q_ltes_in'],
             Q_ltes_out     = res['Q_ltes_out'],
             Q_wtes_in      = res['Q_wtes_in'],
             Q_wtes_out     = res['Q_wtes_out'],
             Q_tes_in       = res['Q_tes_in'],
             Q_tes_out      = res['Q_tes_out'],
             Q_load_in      = res['Q_load_in'],
             PV_prod        = res['PV_prod'],
             Q_hp_th        = res['Q_hp_th'],
             TES_loss       = res['TES_loss'],
             P_THERMAL_LOAD = P_THERMAL_LOAD,
             P_LOAD         = P_LOAD,
             COP_t          = COP_t,
             Cap_frac_t     = res['Cap_frac_t'],
             P_price_buy    = P_price_buy,
             P_price_sell   = P_price_sell,
    )

    opt_json = {k: (float(v)           if isinstance(v, (float, np.floating)) else
                    [float(x) for x in v] if isinstance(v, list)              else v)
                for k, v in opt.items()}
    with open('milp_opt.json', 'w') as f:
        json.dump(opt_json, f, indent=2)

    print("\nResults saved to milp_results.npz and milp_opt.json")

    return opt, res


# ==============================================================================
# --- 6. MAIN EXECUTION ---
# ==============================================================================

if __name__ == "__main__":
    print(f"=== Annual MILP Optimisation (LTES + WTES choice) — {YEAR} ===\n")

    # 1. Load the data
    fetch_pvgis_to_csv(temp_c,temp_h)
    I_SOLAR, COP_t, T_amb     = load_pvgis_from_csv()
    if PRICE_SOURCE == "epex":
        P_price_buy, P_price_sell = load_prices_from_epex(EPEX_CSV_PATH)
    else:
        P_price_buy, P_price_sell = load_prices_from_csv()

    # 2. Calculate the weather-dependent capacity limit using the real temperature
    Cap_frac_t = calculate_capacity_fraction(T_amb)

    # 3. Validation checks
    assert len(P_price_buy) == T, f"Price array length {len(P_price_buy)} != {T}"
    assert len(I_SOLAR)     == T, f"Solar array length {len(I_SOLAR)} != {T}"
    assert len(COP_t)       == T, f"COP array length {len(COP_t)} != {T}"
    assert len(Cap_frac_t)  == T, f"Cap_frac array length {len(Cap_frac_t)} != {T}"

    # 4. Run the optimizer
    opt, res = run_integrated_optimization(I_SOLAR, COP_t, Cap_frac_t, P_price_buy, P_price_sell)