
import pulp
import numpy as np
import pandas as pd
import math
import requests
import json
from Fetch_and_save_data import load_pvgis_from_csv, load_prices_from_csv, fetch_pvgis_to_csv

# ==============================================================================
# --- 1. GLOBAL PARAMETERS & PHYSICAL CONSTANTS ---
# ==============================================================================

YEAR        = 2023
T           = 8760        # hours in a year
time        = np.arange(T)
dt          = 1.0         # hours
BIG_M       = 1e6
MONTHS      = 12
VAT         = 0.21
Factor_Thermal = 0.46

# --- Electrical System ---
CAPEX_BAT_ANNUAL    = 800/20   # €/kWh/yr
CAPEX_PV_ANNUAL     = 900/20   # €/kWp/yr
C_CAP               = 5        # €/kW/month — capacity tariff rate (peak penalty)
ETA_BAT_CH          = 0.95
ETA_BAT_DIS         = 0.95
P_BAT_POWER_RATIO   = 0.5      # kW per kWh
TES_POWER_RATIO     = 0.5      # kW_th per kWh_th — C-rate for both TES types

# --- Design variable upper bounds ---
C_PV_MAX    = 200     # kWp
C_BAT_MAX   = 1000    # kWh
C_HP_MAX    = 300     # kW_th
MAX_ANNUAL_CAPEX = 25000

# --- Heat Pump ---
CAPEX_HP_ANNUAL     = 350.0    # €/kW_th/yr
fixed_cost_annual   = 200      # fixed annual cost of electricity connection
HP_MIN_FRAC         = 0.30     # minimum load fraction when on

# --- NEW: Part-Load & Weather Parameters ---
USE_PARTLOAD        = False     # Set to False for fast linear model, True for part-load real PWL penalty
C_D                 = 0.25     # Degradation coefficient for part-load penalty

# --- PWL Segments (PLR vs EIR) ---
PWL_PLR_BOUNDS = [0.30, 0.60, 1.00] 
PWL_EIR_POINTS = [0.35, 0.60, 1.00]
NUM_PWL_SEGS   = len(PWL_PLR_BOUNDS) - 1

HP_MIN_FRAC    = PWL_PLR_BOUNDS[0]  # minimum load fraction when on

temp_h, temp_c, temp_env = 50, 30, 10   # °C — supply, return, ambient
eta_in,  eta_out         = 0.95, 0.95
dt_sec                   = 3600

# ==============================================================================
# --- 2a. LTES (LATENT THERMAL ENERGY STORAGE) PARAMETERS ---
# ==============================================================================
# PCM: myristic acid (Sharma et al. 2009)
rho_pcm     = 860.0     # kg/m³
L_pcm       = 199.0     # kJ/kg
rho_L_pcm   = rho_pcm * L_pcm / 3600.0   # kWh/m³ ≈ 47.6 kWh/m³

CAPEX_TES_eu_kWh_LTES = 30.0   # €/kWh/yr
LTES_LOSS_FRAC_HR     = 0.005  # 0.5% of total capacity lost per hour (fixed loss per hour)
C_LTES_MAX            = 1000   # kWh_th

# ==============================================================================
# --- 2b. WTES (SENSIBLE / STRATIFIED WATER TANK) PARAMETERS ---
#          Two-zone model: hot zone at T_HIGH, cold zone at T_LOW
# ==============================================================================

T_HIGH  = temp_h   # °C — hot zone (supply temperature)
T_LOW   = temp_c   # °C — cold zone (return temperature)

delta_T_HC  = T_HIGH  - T_LOW          # K — hot-cold spread
delta_T_C0  = T_LOW   - temp_env       # K — cold zone to ambient
delta_T_H0  = T_HIGH  - temp_env       # K — hot zone to ambient

rho_water   = 971.8    # kg/m³
c_water     = 4190.0   # J/(kg·K)

d_wtes      = 1.0                                        # m — inner diameter
A_cross_wtes = math.pi * (d_wtes / 2) ** 2               # m²

U_wall_wtes = 1.0    # W/(m²·K)

kWh_per_m_wtes = (A_cross_wtes * rho_water * c_water * delta_T_HC) / 3.6e6

beta_wtes  = U_wall_wtes * (4.0 / (d_wtes * rho_water * c_water)) * dt_sec / 3.6e6
gamma_wtes = beta_wtes * (delta_T_C0 / delta_T_HC)
loss_lids_wtes = (
    U_wall_wtes * 2.0 * A_cross_wtes
    * ((delta_T_H0 + delta_T_C0) / 2.0)
    * dt_sec / 3.6e6
)

CAPEX_TES_eu_kWh_WTES   = 10.0                                      # €/kWh/yr  
CAPEX_WTES_METER_ANNUAL = CAPEX_TES_eu_kWh_WTES * kWh_per_m_wtes  # €/m/yr

h_wtes_max = 20.0    # m


# ==============================================================================
# --- 3. DEMAND PROFILES ---
# ==============================================================================

GAS_CSV_PATH    = r'C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\gasdata.csv'
GAS_DEMAND_YEAR = 2021

def load_demand_from_gascsv(csv_path=GAS_CSV_PATH, demand_year=GAS_DEMAND_YEAR):
    df_csv = pd.read_csv(
        csv_path,
        sep=';',
        skiprows=1,
        header=None,
        names=['timestamp', 'col2', 'demand_kW_raw', 'extra'],
        decimal=',',
        encoding='utf-8'
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
    print(f"  Raw — mean: {sub.mean():.1f} kW | peak: {sub.max():.1f} kW | total: {sub.sum():.0f} kWh_th")

    if len(sub) == 0:
        raise ValueError(f"No gas data found for year {demand_year}.")

    target_index = pd.date_range(f"{demand_year}-01-01", periods=T, freq='h')
    sub = sub.reindex(target_index)
    n_missing = sub.isna().sum()
    if n_missing > 0:
        print(f"  Warning: {n_missing} missing hours — forward-filled.")
        sub = sub.ffill().bfill()

    thermal    = sub.to_numpy(dtype=float)
    electrical = 90/260 * thermal

    print(f"  Annual profile — mean: {thermal.mean():.1f} kW | peak: {thermal.max():.1f} kW | annual: {thermal.sum():.0f} kWh_th")
    return Factor_Thermal * thermal, electrical


P_THERMAL_LOAD, P_LOAD = load_demand_from_gascsv(GAS_CSV_PATH, GAS_DEMAND_YEAR)

dates_hourly  = pd.date_range(f"{YEAR}-01-01", periods=T, freq='h')
month_of_hour = dates_hourly.month.values - 1   # 0-indexed

# ==============================================================================
# --- 4. HELPER FUNCTIONS ---
# ==============================================================================

def calculate_capacity_fraction(T_amb_array):
    """
    Calculates the weather-dependent capacity limit of the AWHP.
    """
    T_ref = 7.0   # Standard rating temperature
    k = 0.0146    # Degradation slope (Calculated from Daikin 7C vs -15C)
    
    Cap_frac_t = 1.0 + k * (T_amb_array - T_ref)
    Cap_frac_t = np.clip(Cap_frac_t, a_min=0.678, a_max=1.15)
    return Cap_frac_t

# ==============================================================================
# --- 5. INTEGRATED MILP — LTES + WTES CHOICE ---
# ==============================================================================

def run_integrated_optimization(I_SOLAR, COP_t, Cap_frac_t, P_price_buy, P_price_sell):
    prob = pulp.LpProblem("Integrated_MILP_Annual_TES_Choice", pulp.LpMinimize)

    # ------------------------------------------------------------------
    # Design variables
    # ------------------------------------------------------------------
    C_PV   = pulp.LpVariable('C_PV',   lowBound=0, upBound=C_PV_MAX)
    C_bat  = pulp.LpVariable('C_bat',  lowBound=0, upBound=C_BAT_MAX)
    C_HP   = pulp.LpVariable('C_HP',   lowBound=0, upBound=C_HP_MAX)

    C_ltes = pulp.LpVariable('C_ltes', lowBound=0, upBound=C_LTES_MAX)
    h_wtes = pulp.LpVariable('h_wtes', lowBound=0, upBound=h_wtes_max)

    y_PV   = pulp.LpVariable('y_PV',   cat='Binary')
    y_bat  = pulp.LpVariable('y_bat',  cat='Binary')
    y_HP   = pulp.LpVariable('y_HP',   cat='Binary')

    y_ltes = pulp.LpVariable('y_ltes', cat='Binary')
    y_wtes = pulp.LpVariable('y_wtes', cat='Binary')

    P_peak_m = [pulp.LpVariable(f'P_peak_m{m}', lowBound=0) for m in range(MONTHS)]

    # ------------------------------------------------------------------
    # Operational variables (one per hour)
    # ------------------------------------------------------------------
    P_buy      = pulp.LpVariable.dicts('P_buy',     time, lowBound=0)
    P_sell     = pulp.LpVariable.dicts('P_sell',    time, lowBound=0)
    P_bat_ch   = pulp.LpVariable.dicts('P_bat_ch',  time, lowBound=0)
    P_bat_dis  = pulp.LpVariable.dicts('P_bat_dis', time, lowBound=0)
    SoC_bat    = pulp.LpVariable.dicts('SoC_bat',   time, lowBound=0)
    P_hp_elec  = pulp.LpVariable.dicts('P_hp_elec', time, lowBound=0)
    
    # NEW: Variables for HP Output & Part-Load linearization
    q_hp_th      = pulp.LpVariable.dicts('q_hp_th',     time, lowBound=0)
    C_hp_active  = pulp.LpVariable.dicts('C_hp_active', time, lowBound=0)
    
    Q_load_in    = pulp.LpVariable.dicts('Q_load_in',    time, lowBound=0)  
    Q_ltes_in    = pulp.LpVariable.dicts('Q_ltes_in',    time, lowBound=0)  
    Q_ltes_out   = pulp.LpVariable.dicts('Q_ltes_out',   time, lowBound=0)  
    Q_wtes_in    = pulp.LpVariable.dicts('Q_wtes_in',    time, lowBound=0)  
    Q_wtes_out   = pulp.LpVariable.dicts('Q_wtes_out',   time, lowBound=0)
    
    # --- PWL SEGMENT VARIABLES ---
    if USE_PARTLOAD:
        u_hp_seg = pulp.LpVariable.dicts('u_hp_seg', ((t, s) for t in time for s in range(NUM_PWL_SEGS)), cat='Binary')
        q_hp_seg = pulp.LpVariable.dicts('q_hp_seg', ((t, s) for t in time for s in range(NUM_PWL_SEGS)), lowBound=0)
        c_hp_seg = pulp.LpVariable.dicts('c_hp_seg', ((t, s) for t in time for s in range(NUM_PWL_SEGS)), lowBound=0)

    Q_ltes     = pulp.LpVariable.dicts('Q_ltes',    time, lowBound=0)   
    Q_wtes     = pulp.LpVariable.dicts('Q_wtes',    time, lowBound=0)   

    # --- BINARY STATE VARIABLES ---
    u_hp       = pulp.LpVariable.dicts('u_hp',       time, cat='Binary')
    u_bat_ch   = pulp.LpVariable.dicts('u_bat_ch',   time, cat='Binary')
    u_bat_dis  = pulp.LpVariable.dicts('u_bat_dis',  time, cat='Binary')
    u_ltes_in  = pulp.LpVariable.dicts('u_ltes_in',  time, cat='Binary')
    u_ltes_out = pulp.LpVariable.dicts('u_ltes_out', time, cat='Binary')
    u_wtes_in  = pulp.LpVariable.dicts('u_wtes_in',  time, cat='Binary')
    u_wtes_out = pulp.LpVariable.dicts('u_wtes_out', time, cat='Binary')

    # ------------------------------------------------------------------
    # Affine capacity expressions
    # ------------------------------------------------------------------
    C_WTES = kWh_per_m_wtes * h_wtes   

    # ------------------------------------------------------------------
    # Objective: minimise total annualised cost
    # ------------------------------------------------------------------
    inv_cost = (CAPEX_PV_ANNUAL       * C_PV
                + CAPEX_BAT_ANNUAL    * C_bat
                + CAPEX_HP_ANNUAL     * C_HP
                + CAPEX_TES_eu_kWh_LTES * C_ltes
                + CAPEX_WTES_METER_ANNUAL * h_wtes)

    op_buy     = pulp.lpSum(P_price_buy[t]  * P_buy[t]  * dt for t in time)
    op_sell    = pulp.lpSum(P_price_sell[t] * P_sell[t] * dt for t in time)
    cap_tariff = pulp.lpSum(C_CAP * P_peak_m[m] for m in range(MONTHS))

    prob += (inv_cost
             + (op_buy - op_sell + cap_tariff + fixed_cost_annual) * (1 + VAT),
             "Total_Annual_Cost")
    prob += inv_cost <= MAX_ANNUAL_CAPEX, "Max_Annualized_CAPEX"
    # ------------------------------------------------------------------
    # Technology selection constraints
    # ------------------------------------------------------------------
    prob += C_PV  <= C_PV_MAX  * y_PV,  "Cap_PV"
    prob += C_bat <= C_BAT_MAX * y_bat, "Cap_bat"
    prob += C_HP  <= C_HP_MAX  * y_HP,  "Cap_HP"

    prob += y_ltes + y_wtes == 1, "TES_type_select_one"
    prob += C_ltes <= C_LTES_MAX * y_ltes, "LTES_select"
    prob += h_wtes <= h_wtes_max * y_wtes, "WTES_select"

    # ------------------------------------------------------------------
    # Hourly constraints
    # ------------------------------------------------------------------
    print("Building hourly constraints (8 760 steps) …")
    for t in time:
        if t % 1000 == 0:
            print(f"  … step {t}/{T}")
        prev = T - 1 if t == 0 else t - 1
        m    = int(month_of_hour[t])

        PV_t      = I_SOLAR[t]   * C_PV
     
        # ------------------------------------------------------------------
        # Heat Pump Operation Logic
        # ------------------------------------------------------------------
        if USE_PARTLOAD:
            prob += C_hp_active[t] <= C_HP_MAX * u_hp[t], f"HPAct_Zero_{t}"
            prob += C_hp_active[t] <= C_HP, f"HPAct_Cap_{t}"
            prob += C_hp_active[t] >= C_HP - C_HP_MAX * (1 - u_hp[t]), f"HPAct_Match_{t}"

            prob += pulp.lpSum(u_hp_seg[t, s] for s in range(NUM_PWL_SEGS)) == u_hp[t], f"HPSeg_Active_{t}"
            prob += pulp.lpSum(c_hp_seg[t, s] for s in range(NUM_PWL_SEGS)) == C_hp_active[t], f"HPSeg_Cap_Sum_{t}"

            p_elec_total_expr = 0
            q_th_total_expr = 0

            for s in range(NUM_PWL_SEGS):
                prob += c_hp_seg[t, s] <= C_HP_MAX * u_hp_seg[t, s], f"HPSeg_Cap_Max_{t}_{s}"
                
                plr_min = PWL_PLR_BOUNDS[s]
                plr_max = PWL_PLR_BOUNDS[s+1]
                eir_min = PWL_EIR_POINTS[s]
                eir_max = PWL_EIR_POINTS[s+1]
                
                slope = (eir_max - eir_min) / (plr_max - plr_min)
                intercept = eir_min - slope * plr_min
                
                cap_t = c_hp_seg[t, s] * Cap_frac_t[t]
                
                prob += q_hp_seg[t, s] >= plr_min * cap_t, f"HPSeg_QMin_{t}_{s}"
                prob += q_hp_seg[t, s] <= plr_max * cap_t, f"HPSeg_QMax_{t}_{s}"
                
                q_th_total_expr += q_hp_seg[t, s]
                
                p_elec_total_expr += (slope * q_hp_seg[t, s]) + (intercept * cap_t)

            prob += q_hp_th[t] == q_th_total_expr, f"HP_Q_Total_{t}"
            prob += P_hp_elec[t] * COP_t[t] == p_elec_total_expr, f"HP_P_Total_{t}"

        else:
            prob += C_hp_active[t] <= C_HP_MAX * u_hp[t], f"HPAct_Zero_{t}"
            prob += C_hp_active[t] <= C_HP, f"HPAct_Cap_{t}"
            prob += C_hp_active[t] >= C_HP - C_HP_MAX * (1 - u_hp[t]), f"HPAct_Match_{t}"

            prob += q_hp_th[t] <= C_hp_active[t] * Cap_frac_t[t], f"HPMax_Weather_{t}"
            prob += q_hp_th[t] >= HP_MIN_FRAC * C_hp_active[t] * Cap_frac_t[t], f"HPMin_Weather_{t}"
            
            prob += P_hp_elec[t] * COP_t[t] == (C_D * C_hp_active[t] * Cap_frac_t[t]) + ((1 - C_D) * q_hp_th[t]), f"HPPenalty_{t}"
            
        # ------------------------------------------------------------------

        # --- Electrical balance ---
        prob += (PV_t + P_bat_dis[t] + P_buy[t]
                 == P_LOAD[t] + P_bat_ch[t] + P_hp_elec[t] + P_sell[t],
                 f"Elec_{t}")

        prob += P_sell[t] <= PV_t + P_bat_dis[t], f"SellMax_{t}"

        # --- Battery dynamics & Mutually Exclusive Flow Limits ---
        prob += u_bat_ch[t] + u_bat_dis[t] <= 1, f"BatExcl_{t}"
        prob += P_bat_ch[t]  <= P_BAT_POWER_RATIO * C_BAT_MAX * u_bat_ch[t], f"BatChMax_{t}"
        prob += P_bat_dis[t] <= P_BAT_POWER_RATIO * C_BAT_MAX * u_bat_dis[t], f"BatDisMax_{t}"
        prob += P_bat_ch[t]  <= P_BAT_POWER_RATIO * C_bat, f"BatChCap_{t}"
        prob += P_bat_dis[t] <= P_BAT_POWER_RATIO * C_bat, f"BatDisCap_{t}"
        prob += SoC_bat[t]   <= C_bat,                       f"SoCMax_{t}"
        prob += (SoC_bat[t]  == SoC_bat[prev]
                 + (ETA_BAT_CH * P_bat_ch[t]
                    - (1.0 / ETA_BAT_DIS) * P_bat_dis[t]) * dt,
                 f"SoCDyn_{t}")

        # ---- LTES dynamics (Fixed fractional hourly loss based on capacity) ----
        loss_ltes_t = LTES_LOSS_FRAC_HR * C_ltes

        prob += (Q_ltes[t] == Q_ltes[prev]
                 - loss_ltes_t
                 + Q_ltes_in[t] - Q_ltes_out[t],
                 f"LTESDyn_{t}")

        # LTES Mutually Exclusive Flow Limits
        prob += u_ltes_in[t] + u_ltes_out[t] <= y_ltes, f"LTESExcl_{t}"
        prob += Q_ltes_in[t]  <= BIG_M * u_ltes_in[t], f"LTESInBin_{t}"
        prob += Q_ltes_out[t] <= BIG_M * u_ltes_out[t], f"LTESOutBin_{t}"
        prob += Q_ltes_in[t]  <= TES_POWER_RATIO * C_ltes, f"LTESInCap_{t}"
        prob += Q_ltes_out[t] <= TES_POWER_RATIO * C_ltes, f"LTESOutCap_{t}"

        # LTES SoC bounds
        prob += Q_ltes[t] >= 0.05 * C_ltes - BIG_M * (1 - y_ltes), f"LTESMin_{t}"
        prob += Q_ltes[t] <= 0.95 * C_ltes + BIG_M * (1 - y_ltes), f"LTESMax_{t}"
        prob += Q_ltes[t] <= BIG_M * y_ltes, f"LTESZero_{t}"

        # ---- WTES dynamics (Original geometry/temperature-based loss) ---------
        loss_wtes_t = (beta_wtes  * Q_wtes[prev]
                       + gamma_wtes * C_WTES
                       + loss_lids_wtes * y_wtes)

        prob += (Q_wtes[t] == Q_wtes[prev]
                 - loss_wtes_t
                 + Q_wtes_in[t] - Q_wtes_out[t],
                 f"WTESDyn_{t}")

        # WTES Mutually Exclusive Flow Limits
        prob += u_wtes_in[t] + u_wtes_out[t] <= y_wtes, f"WTESExcl_{t}"
        prob += Q_wtes_in[t]  <= BIG_M * u_wtes_in[t], f"WTESInBin_{t}"
        prob += Q_wtes_out[t] <= BIG_M * u_wtes_out[t], f"WTESOutBin_{t}"
        prob += Q_wtes_in[t]  <= TES_POWER_RATIO * C_WTES, f"WTESInCap_{t}"
        prob += Q_wtes_out[t] <= TES_POWER_RATIO * C_WTES, f"WTESOutCap_{t}"

        # WTES SoC bounds
        prob += Q_wtes[t] >= 0.05 * C_WTES - BIG_M * (1 - y_wtes), f"WTESMin_{t}"
        prob += Q_wtes[t] <= 0.95 * C_WTES + BIG_M * (1 - y_wtes), f"WTESMax_{t}"
        prob += Q_wtes[t] <= BIG_M * y_wtes, f"WTESZero_{t}"

        # ---- Thermal balance (parallel topology) -------------------------
        prob += (q_hp_th[t] == Q_ltes_in[t] + Q_wtes_in[t] + Q_load_in[t],
                 f"HPSplit_{t}")

        prob += (Q_load_in[t] + Q_ltes_out[t] + Q_wtes_out[t] == P_THERMAL_LOAD[t],
                 f"ThermLoad_{t}")

        # --- Monthly peak tracking ---
        prob += P_peak_m[m] >= P_buy[t], f"PeakM_{t}"


    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    print("\nSolving (this may take several minutes for 8 760-step MILP) …")
    prob.solve(pulp.GUROBI(msg=1, gapRel=0.01, timeLimit=200))
    status = pulp.LpStatus[prob.status]
    print(f"\nSolver status: {status}")

    # ------------------------------------------------------------------
    # Extract results
    # ------------------------------------------------------------------
    y_ltes_val = int(round(pulp.value(y_ltes)))
    y_wtes_val = int(round(pulp.value(y_wtes)))
    tes_type   = "LTES (PCM)" if y_ltes_val else "WTES (stratified water)"

    C_LTES_val   = pulp.value(C_ltes)
    V_ltes_val   = C_LTES_val / rho_L_pcm if rho_L_pcm else 0.0
    mass_pcm     = rho_pcm * V_ltes_val

    h_wtes_val   = pulp.value(h_wtes)
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
        'C_PV':       pulp.value(C_PV),
        'C_bat':      pulp.value(C_bat),
        'C_HP':       pulp.value(C_HP),
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
        'P_peak_m':   [pulp.value(P_peak_m[m]) for m in range(MONTHS)],
        'obj':        pulp.value(prob.objective),
    }
    opt['P_peak_annual'] = max(opt['P_peak_m'])

    arr = lambda d: np.array([pulp.value(d[t]) for t in time])
    Q_tes_active = Q_ltes if y_ltes_val else Q_wtes

    Q_tes_in_combined  = np.array([pulp.value(Q_ltes_in[t])  + pulp.value(Q_wtes_in[t])  for t in time])
    Q_tes_out_combined = np.array([pulp.value(Q_ltes_out[t]) + pulp.value(Q_wtes_out[t]) for t in time])

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
        res['TES_loss'] = np.array([
            LTES_LOSS_FRAC_HR * C_LTES_val
            for t in time
        ])
    else:
        res['TES_loss'] = np.array([
            beta_wtes * res['Q_tes'][T-1 if t == 0 else t-1]
            + gamma_wtes * C_WTES_val
            + loss_lids_wtes
            for t in time
        ])

    capex_pv   = CAPEX_PV_ANNUAL           * opt['C_PV']
    capex_bat  = CAPEX_BAT_ANNUAL          * opt['C_bat']
    capex_hp   = CAPEX_HP_ANNUAL           * opt['C_HP']
    capex_ltes = CAPEX_TES_eu_kWh_LTES     * opt['C_ltes']
    capex_wtes = CAPEX_WTES_METER_ANNUAL   * opt['h_wtes']
    capex_tes  = capex_ltes + capex_wtes   
    capex_total = capex_pv + capex_bat + capex_hp + capex_tes

    opex_buy      = sum(P_price_buy[t]  * res['P_buy'][t]  * dt for t in time)
    opex_sell     = sum(P_price_sell[t] * res['P_sell'][t] * dt for t in time)
    opex_cap_tar  = sum(C_CAP * pm for pm in opt['P_peak_m'])
    opex_fixed    = fixed_cost_annual
    opex_net      = (opex_buy - opex_sell + opex_cap_tar + opex_fixed) * (1 + VAT)

    print(f"\n=== OPTIMISATION RESULTS ===")
    print(f"  PV:             {opt['C_PV']:.2f} kWp")
    print(f"  Battery:        {opt['C_bat']:.2f} kWh")
    print(f"  HP:             {opt['C_HP']:.2f} kW_th")
    print(f"  TES selected:   {tes_type}")

    if y_ltes_val:
        print(f"  LTES tank:      C_ltes  = {opt['C_ltes']:.2f} kWh_th")
        print(f"                  V_ltes  = {opt['V_ltes']:.3f} m³  ({opt['V_ltes']*1000:.0f} L)")
        print(f"                  m_PCM   = {opt['mass_pcm']:.0f} kg  (myristic acid)")
    else:
        print(f"  WTES tank:      h_wtes  = {opt['h_wtes']:.3f} m")
        print(f"                  V_wtes  = {opt['V_wtes']:.3f} m³  ({opt['V_wtes']*1000:.0f} L)")
        print(f"                  m_water = {opt['mass_water']:.0f} kg")
        print(f"                  C_TES   = {opt['C_WTES']:.2f} kWh_th  (sensible, {T_LOW}–{T_HIGH} °C)")

    print(f"\n--- CAPEX (annualised) ---")
    print(f"  PV:             €{capex_pv:>10,.0f}/yr")
    print(f"  Battery:        €{capex_bat:>10,.0f}/yr")
    print(f"  Heat Pump:      €{capex_hp:>10,.0f}/yr")
    if y_ltes_val:
        print(f"  LTES:           €{capex_ltes:>10,.0f}/yr  ({opt['C_ltes']:.2f} kWh × {CAPEX_TES_eu_kWh_LTES:.2f} €/kWh/yr)")
    else:
        print(f"  WTES:           €{capex_wtes:>10,.0f}/yr  ({opt['h_wtes']:.2f} m × {CAPEX_WTES_METER_ANNUAL:.2f} €/m/yr)")
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