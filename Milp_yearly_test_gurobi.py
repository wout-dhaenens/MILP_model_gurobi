import os
import calendar
import numpy as np
import pandas as pd
import math
import requests
import json
import gurobipy as gp  # 2. Now it will find your unrestricted academic license
from gurobipy import GRB
import fetch_solcast_data as _scd
from Fetch_and_save_data import (load_prices_from_csv, load_prices_from_epex,
                                  fetch_prices_to_csv)
from demand_generation import generate_yearly_bdew_profile, load_bdew_demand_from_csv

# ==============================================================================
# --- 1. GLOBAL PARAMETERS & PHYSICAL CONSTANTS ---
# ==============================================================================

# --- Year configuration ---
# WEATHER_YEAR controls: solar irradiance (PVGIS), COP, and thermal demand (BDEW)
# PRICE_YEAR   controls: electricity prices (ENTSO-E day-ahead)
WEATHER_YEAR = 2025
PRICE_YEAR   = 2025

YEAR        = WEATHER_YEAR   # backward compat (dates_hourly etc.)
T           = 8784 if calendar.isleap(WEATHER_YEAR) else 8760  # auto leap-year

# --- Price source selection ---
# "csv"  : load from prices_data.csv (fetched via Entsoe API, using PRICE_YEAR)
# "epex" : load from epex_2025.csv   (downloaded EPEX Belgium day-ahead 2025)
PRICE_SOURCE   = "csv"
EPEX_CSV_PATH  = "epex_2025.csv"
dt          = 1.0         # hours
BIG_M       = 1e6
MONTHS      = 12
VAT         = 0.21

# --- Discount rate & technology lifetimes ---
DISCOUNT_RATE   = 0.05   # [-]   weighted average cost of capital
LIFETIME_PV     = 25     # yr
LIFETIME_BAT    = 15     # yr
LIFETIME_HP     = 20     # yr
LIFETIME_TES    = 20     # yr

def crf(r, n):
    """Capital Recovery Factor — annualises a one-time CAPEX over n years at rate r."""
    if r == 0:
        return 1.0 / n
    return r * (1 + r)**n / ((1 + r)**n - 1)

# --- One-time CAPEX (€/unit) ---
# Hardware costs fitted from market data; installation from invoice (15x410Wp, pitched roof)
# Installation: labour €70/panel + mounting €40/panel + 50% inverter placement €225 + inspection €150
BAT_CAPEX_FIXED = 848.0    # € one-time fixed component  (hardware 623 + installation 225)
BAT_CAPEX_VAR   = 456.0    # €/kWh one-time variable component  (hardware only, no variable install)
PV_CAPEX_FIXED  = 919.0    # € one-time fixed component  (hardware 544 + installation 375)
PV_CAPEX_VAR    = 532.0    # €/kWp one-time variable component  (hardware 264 + installation 268)
HP_CAPEX_FIXED  = 9516.0   # € one-time fixed component  (9,516/Q_hp + 644 €/kW formula)
HP_CAPEX_VAR    = 644.0    # €/kW_th one-time variable component
LTES_CAPEX_FIXED       = 721.0   # € one-time fixed component  (fitted: 721 + 197*kWh)
LTES_CAPEX_VAR_PER_KWH = 197.0   # €/kWh_th one-time variable component

# --- Electrical System ---
BAT_CAPEX_FIXED_ANNUAL = BAT_CAPEX_FIXED * crf(DISCOUNT_RATE, LIFETIME_BAT)  # €/yr
BAT_CAPEX_VAR_ANNUAL   = BAT_CAPEX_VAR   * crf(DISCOUNT_RATE, LIFETIME_BAT)  # €/kWh/yr
PV_CAPEX_FIXED_ANNUAL = PV_CAPEX_FIXED * crf(DISCOUNT_RATE, LIFETIME_PV)  # €/yr
PV_CAPEX_VAR_ANNUAL   = PV_CAPEX_VAR   * crf(DISCOUNT_RATE, LIFETIME_PV)  # €/kWp/yr
C_CAP               = 5        # €/kW/month
ETA_BAT_CH          = 0.95
ETA_BAT_DIS         = 0.95
ETA_TES_CH          = 0.95     # charge efficiency (HP heat → stored heat)
ETA_TES_DIS         = 0.95     # discharge efficiency (stored heat → delivered heat)
P_BAT_POWER_RATIO   = 0.5      # kW per kWh
TES_POWER_RATIO     = 0.5      # kW_th per kWh_th

# --- Design variable upper bounds ---
C_PV_MAX        = 50     # kWp
C_BAT_MAX       = 200    # kWh
C_HP_MAX        = 200    # kW_th
MAX_ANNUAL_CAPEX = 250000       # €/yr — Maximum allowed annualized CAPEX
MAX_VOLUME_TES  = 20.0   # m³ — maximum physical volume for either TES type
                          #       LTES: C_LTES_MAX  = MAX_VOLUME_TES × LTES_ENERGY_DENSITY [kWh_th]
                          #       WTES: h_wtes_max  = MAX_VOLUME_TES / A_cross_wtes         [m]

# --- Heat Pump ---
CAPEX_HP_FIXED_ANNUAL = HP_CAPEX_FIXED * crf(DISCOUNT_RATE, LIFETIME_HP)  # €/yr  (fixed part, if installed)
CAPEX_HP_VAR_ANNUAL   = HP_CAPEX_VAR   * crf(DISCOUNT_RATE, LIFETIME_HP)  # €/kW_th/yr (variable part)
fixed_cost_annual   = 200      
HP_MIN_FRAC         = 0.222     

C_D                 = 0.10
HP_MIN_FRAC         = 0.222    # 22.2% minimum load
ETA_LORENZ          = 0.361    # η_Lorenz: COP_actual / COP_Lorenz_ideal (calibrated from EWYE050CZNAA2 data)

temp_h, temp_c, temp_env = 60, 40, 20   # °C — supply, return, ambient
dt_sec                   = 3600

# ==============================================================================
# --- 2a. LTES (LATENT THERMAL ENERGY STORAGE) PARAMETERS ---
# ==============================================================================
rho_pcm     = 860.0     # kg/m³
L_pcm       = 199.0     # kJ/kg
rho_L_pcm   = rho_pcm * L_pcm / 3600.0   # kWh/m³ ≈ 47.6 kWh/m³

LTES_CAPEX_FIXED_ANNUAL  = LTES_CAPEX_FIXED       * crf(DISCOUNT_RATE, LIFETIME_TES)  # €/yr
LTES_CAPEX_VAR_ANNUAL    = LTES_CAPEX_VAR_PER_KWH * crf(DISCOUNT_RATE, LIFETIME_TES)  # €/kWh_th/yr
LTES_LOSS_FIXED_HR    = 0.016667  # kWh/h — fixed standby loss when LTES is installed (fitted from Sunamp Thermino e)
LTES_LOSS_FRAC_HR     = 0.001393  # /h    — proportional standby loss per kWh_th capacity (fitted from Sunamp Thermino e)
LTES_ENERGY_DENSITY   = 50.0     # kWh/m³ (= 0.05 kWh/L) — PCM energy density for capacity bound

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

d_wtes      = 1.75                                        
A_cross_wtes = math.pi * (d_wtes / 2) ** 2               

U_wall_wtes = 0.4    

kWh_per_m_wtes = (A_cross_wtes * rho_water * c_water * delta_T_HC) / 3.6e6

beta_wtes  = U_wall_wtes * (4.0 / (d_wtes * rho_water * c_water)) * dt_sec
gamma_wtes = beta_wtes * (delta_T_C0 / delta_T_HC)
loss_lids_wtes = (
    U_wall_wtes * 2.0 * A_cross_wtes
    * ((delta_T_H0 + delta_T_C0) / 2.0)
    * dt_sec / 3.6e6
)

# WTES CAPEX — fitted from market data (Vaillant, Bosch, Viessmann, Remeha; V >= 100 L)
# Fit: Total one-time CAPEX = 921 + 1.165 * V_L  [€]  (R² = 0.85)
# Converted to kWh_th using kWh_per_L = rho*c*dT / 3.6e6:
#   Total one-time CAPEX = WTES_CAPEX_FIXED + WTES_CAPEX_VAR_PER_KWH * V_kWh  [€]

kWh_per_L_wtes          = (rho_water / 1000.0) * c_water * delta_T_HC / 3.6e6  # kWh/L
WTES_CAPEX_FIXED        = 921.0                                      # € one-time fixed component
WTES_CAPEX_VAR_PER_L    = 1.165                                    # €/L one-time variable component
WTES_CAPEX_VAR_PER_KWH  = WTES_CAPEX_VAR_PER_L / kWh_per_L_wtes    # €/kWh_th one-time

WTES_CAPEX_FIXED_ANNUAL  = WTES_CAPEX_FIXED       * crf(DISCOUNT_RATE, LIFETIME_TES)  # €/yr
WTES_CAPEX_VAR_ANNUAL    = WTES_CAPEX_VAR_PER_KWH * crf(DISCOUNT_RATE, LIFETIME_TES)  # €/kWh_th/yr

h_wtes_max  = MAX_VOLUME_TES / A_cross_wtes                          # m      — derived from shared volume limit
C_LTES_MAX  = MAX_VOLUME_TES * LTES_ENERGY_DENSITY                  # kWh_th — derived from shared volume limit
C_WTES_MAX  = kWh_per_m_wtes * h_wtes_max                           # kWh_th — max WTES capacity
WTES_REF_KWH = 500.0 * kWh_per_L_wtes                               # kWh_th at 500 L reference
# ==============================================================================
# --- PWL CAPEX PARAMETER GENERATOR ---
# ==============================================================================
NUM_CAPEX_SEGS = 3  # You can adjust this to 4, 5, 6, etc.

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

# PV CAPEX is linear (no PWL needed): cost_pv = PV_CAPEX_FIXED_ANNUAL*y_PV + PV_CAPEX_VAR_ANNUAL*C_PV
# BAT CAPEX is linear (no PWL needed): cost_bat = BAT_CAPEX_FIXED_ANNUAL*y_bat + BAT_CAPEX_VAR_ANNUAL*C_bat
# HP CAPEX is linear (no PWL needed): cost_hp = CAPEX_HP_FIXED_ANNUAL*y_HP + CAPEX_HP_VAR_ANNUAL*C_HP
# LTES CAPEX is linear (no PWL needed): cost_ltes = LTES_CAPEX_FIXED_ANNUAL*y_ltes + LTES_CAPEX_VAR_ANNUAL*C_ltes
# WTES CAPEX is linear (no PWL needed): cost_wtes = WTES_CAPEX_FIXED_ANNUAL*y_wtes + WTES_CAPEX_VAR_ANNUAL*C_WTES

# ==============================================================================
# --- 3. DEMAND PROFILES ---
# ==============================================================================
# Thermal demand is generated via BDEW model using weather data for WEATHER_YEAR.
# The CSV is regenerated when the main script runs (see __main__).
# P_THERMAL_LOAD is set in __main__ after generating/loading the BDEW profile.

# --- Measured electricity demand from smart meters ---
METER_CSV_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "combined_grid_data.csv")
ELEC_DEMAND_YEAR   = 2025

def load_electrical_demand_from_meter(csv_path=METER_CSV_PATH, demand_year=ELEC_DEMAND_YEAR):
    """
    Load real measured total building electricity demand (Totaal_verbruik_kWh =
    grid offtake + PV production - grid injection) from combined_grid_data.csv,
    resample from 15-min to hourly, and return as a kW array of length 8760.
    kWh per 15-min summed to hourly kWh == average kW over that hour.
    """
    df = pd.read_csv(csv_path, sep=";", decimal=",",
                     parse_dates=["timestamp"], index_col="timestamp")

    # Keep only the selected year — use total building demand (incl. PV self-consumption)
    df = df[df.index.year == demand_year]["Totaal_verbruik_kWh"]

    # Sum 15-min kWh to hourly kWh → equals average kW
    hourly = df.resample("h").sum()

    # Align to a full-year index (fill any gaps with forward-fill)
    target_index = pd.date_range(f"{demand_year}-01-01", periods=T, freq="h")
    hourly = hourly.reindex(target_index)
    if hourly.isna().sum() > 0:
        print(f"  Warning: {hourly.isna().sum()} missing hours filled by interpolation.")
        hourly = hourly.interpolate(method="time").ffill().bfill()

    print(f"Meter CSV loaded: {len(hourly)} hourly records for {demand_year} "
          f"| Total = {hourly.sum():.1f} kWh = {hourly.sum()/1000:.2f} MWh")
    return hourly.to_numpy(dtype=float)

P_LOAD = load_electrical_demand_from_meter(METER_CSV_PATH, ELEC_DEMAND_YEAR)

dates_hourly  = pd.date_range(f"{YEAR}-01-01", periods=T, freq='h')
month_of_hour = dates_hourly.month.values - 1   

# ==============================================================================
# --- 4. HELPER FUNCTIONS ---
# ==============================================================================
def calculate_capacity_fraction(T_amb_array):
    # Fitted from EWYE050CZNAA2 manufacturer simulation data:
    #   - 8 heating points at T_amb = -15 to +7°C (full load, CLWT=55°C)
    #   - 1 high-T point at T_amb = +20°C from separate simulation
    #   - Forced through (T_ref=7°C, Cap_frac=1.0), R²=0.997
    # k = 0.02106  (+44% vs old literature value of 0.0146)
    # Clips:
    #   a_min = 0.5366 at -15°C (old: 0.678)
    #   a_max = 1.2738 at +20°C (old: 1.15)
    T_ref = 7.0
    k     = 0.02106
    Cap_frac_t = 1.0 + k * (T_amb_array - T_ref)
    return np.clip(Cap_frac_t, a_min=0.5366, a_max=1.2738)

# ==============================================================================
# --- 5. INTEGRATED MILP (GUROBIPY) ---
# ==============================================================================
def run_integrated_optimization(I_SOLAR, COP_t, Cap_frac_t, P_price_buy, P_price_sell):
    model = gp.Model("Integrated_MILP_Annual_TES_Choice")

    # ------------------------------------------------------------------
    # Design variables (Scalars)
    # ------------------------------------------------------------------
    C_PV   = model.addVar(lb=0, ub=C_PV_MAX, name="C_PV")
    C_bat  = model.addVar(lb=0, ub=C_BAT_MAX, name="C_bat")
    C_HP   = model.addVar(lb=0, ub=C_HP_MAX, name="C_HP")
    C_ltes = model.addVar(lb=0, ub=C_LTES_MAX, name="C_ltes")
    h_wtes     = model.addVar(lb=0, ub=h_wtes_max, name="h_wtes")
    C_WTES_var = model.addVar(lb=0, ub=C_WTES_MAX, name="C_WTES_var")

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
    
    Q_ltes     = model.addVars(T, lb=0, name="Q_ltes")
    Q_wtes     = model.addVars(T, lb=0, name="Q_wtes")

    u_hp = model.addVars(T, vtype=GRB.BINARY, name="u_hp")

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

    # PV CAPEX: fitted from market data (greenakku.de + solarwinkel.be) → Total = 544 + 264*kWp [€]
    model.addConstr(cost_pv == PV_CAPEX_FIXED_ANNUAL * y_PV + PV_CAPEX_VAR_ANNUAL * C_PV, "PV_CAPEX_linear")
    # BAT CAPEX: fitted from market data (BYD/SolarEdge/Huawei) → Total = 623 + 456*kWh [€]
    model.addConstr(cost_bat == BAT_CAPEX_FIXED_ANNUAL * y_bat + BAT_CAPEX_VAR_ANNUAL * C_bat, "BAT_CAPEX_linear")
    # HP CAPEX: specific cost = 9516/Q_hp + 644 €/kW  →  total = 9516 + 644*Q_hp [€]
    model.addConstr(cost_hp == CAPEX_HP_FIXED_ANNUAL * y_HP + CAPEX_HP_VAR_ANNUAL * C_HP, "HP_CAPEX_linear")
    # LTES CAPEX: fitted from market data (Flamco, Sunamp, Kraftbox) → Total = 721 + 197*kWh [€]
    model.addConstr(cost_ltes == LTES_CAPEX_FIXED_ANNUAL * y_ltes + LTES_CAPEX_VAR_ANNUAL * C_ltes, "LTES_CAPEX_linear")
    model.addConstr(C_WTES_var == kWh_per_m_wtes * h_wtes, "WTES_kWh_link")
    # WTES CAPEX: fitted from market data → Total = 921 + 1.165*V_L = FIXED + VAR*V_kWh [€]
    model.addConstr(cost_wtes == WTES_CAPEX_FIXED_ANNUAL * y_wtes + WTES_CAPEX_VAR_ANNUAL * C_WTES_var, "WTES_CAPEX_linear")

    # ------------------------------------------------------------------
    # Hourly constraints
    # ------------------------------------------------------------------
    print("Building hourly constraints (8 760 steps) …")
    for t in range(T):
        prev = T - 1 if t == 0 else t - 1
        m    = int(month_of_hour[t])

        PV_t = I_SOLAR[t] * C_PV
     
        # --- Heat Pump Operation Logic ---
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
        model.addConstr(P_bat_ch[t]  <= P_BAT_POWER_RATIO * C_bat, f"BatChCap_{t}")
        model.addConstr(P_bat_dis[t] <= P_BAT_POWER_RATIO * C_bat, f"BatDisCap_{t}")
        model.addConstr(SoC_bat[t]   <= C_bat, f"SoCMax_{t}")
        model.addConstr(SoC_bat[t]   == SoC_bat[prev] + (ETA_BAT_CH * P_bat_ch[t] - (1.0 / ETA_BAT_DIS) * P_bat_dis[t]) * dt, f"SoCDyn_{t}")

        # ---- LTES dynamics ----
        # Standby loss = fixed term (when installed) + proportional term (per kWh_th capacity)
        loss_ltes_t = LTES_LOSS_FIXED_HR * y_ltes + LTES_LOSS_FRAC_HR * C_ltes

        model.addConstr(Q_ltes[t] == Q_ltes[prev] - loss_ltes_t + ETA_TES_CH * Q_ltes_in[t] - (1.0 / ETA_TES_DIS) * Q_ltes_out[t], f"LTESDyn_{t}")
        model.addConstr(Q_ltes_in[t]  <= TES_POWER_RATIO * C_ltes, f"LTESInCap_{t}")
        model.addConstr(Q_ltes_out[t] <= TES_POWER_RATIO * C_ltes, f"LTESOutCap_{t}")
        model.addConstr(Q_ltes[t] >= 0.05 * C_ltes - BIG_M * (1 - y_ltes), f"LTESMin_{t}")
        model.addConstr(Q_ltes[t] <= 0.95 * C_ltes + BIG_M * (1 - y_ltes), f"LTESMax_{t}")
        model.addConstr(Q_ltes[t] <= BIG_M * y_ltes, f"LTESZero_{t}")

        # ---- WTES dynamics ----
        loss_wtes_t = (beta_wtes * Q_wtes[prev] + gamma_wtes * C_WTES + loss_lids_wtes * y_wtes)

        model.addConstr(Q_wtes[t] == Q_wtes[prev] - loss_wtes_t + ETA_TES_CH * Q_wtes_in[t] - (1.0 / ETA_TES_DIS) * Q_wtes_out[t], f"WTESDyn_{t}")
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
    model.setParam('MIPGap', 0.03)
    model.setParam('TimeLimit', 600)
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
        res['TES_loss'] = np.array([LTES_LOSS_FIXED_HR + LTES_LOSS_FRAC_HR * C_LTES_val for t in range(T)])
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
        capex_ltes_eu_kwh = capex_ltes / opt['C_ltes'] if opt['C_ltes'] > 0 else 0.0
        print(f"  LTES:           €{capex_ltes:>10,.0f}/yr  ({capex_ltes_eu_kwh:.2f} €/kWh/yr)")
    else:
        capex_wtes_eu_kwh = capex_wtes / opt['C_WTES'] if opt['C_WTES'] > 0 else 0.0
        print(f"  WTES:           €{capex_wtes:>10,.0f}/yr  ({capex_wtes_eu_kwh:.2f} €/kWh/yr)")
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
    print(f"  Peak thermal load:  {max(P_THERMAL_LOAD):.2f} kW_th")

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
    print(f"=== Annual MILP Optimisation (LTES + WTES choice) ===")
    print(f"    Weather/COP/demand year : {WEATHER_YEAR}")
    print(f"    Electricity price year  : {PRICE_YEAR}")
    print(f"    Price source            : {PRICE_SOURCE}")
    print(f"    η_Lorenz (ETA_LORENZ)   : {ETA_LORENZ}\n")

    # 1a. Load solar + temperature data for WEATHER_YEAR via Solcast
    #     → reads from CSV if it already exists, otherwise fetches from the API (12 calls)
    _scd.YEAR       = WEATHER_YEAR
    _scd.T          = T
    _scd.ETA_LORENZ = ETA_LORENZ   # propagate calibrated η before any fetch/recompute

    solcast_csv = f"solcast_data_{WEATHER_YEAR}.csv"
    if os.path.exists(solcast_csv):
        print(f"  [Solcast] CSV found → loading from '{solcast_csv}' (no API call)")
        I_SOLAR, _, T_amb = _scd.load_solcast_from_csv(solcast_csv)
    else:
        print(f"  [Solcast] CSV not found → fetching from API for {WEATHER_YEAR}...")
        _scd.fetch_solcast_to_csv(temp_c=temp_c, temp_h=temp_h, year=WEATHER_YEAR)
        I_SOLAR, _, T_amb = _scd.load_solcast_from_csv(solcast_csv)

    # Always recompute COP with the current ETA_LORENZ / temp_c / temp_h
    # (overrides whatever was cached in the CSV)
    COP_t = np.array([_scd.calculate_lorenz_cop(temp_c, temp_h, t) for t in T_amb])

    # 1b. Fetch electricity prices for PRICE_YEAR
    if PRICE_SOURCE == "epex":
        P_price_buy, P_price_sell = load_prices_from_epex(EPEX_CSV_PATH)
    else:
        fetch_prices_to_csv(year=PRICE_YEAR)
        P_price_buy, P_price_sell = load_prices_from_csv()

    # 1c. Generate BDEW thermal demand for WEATHER_YEAR
    generate_yearly_bdew_profile(WEATHER_YEAR)
    P_THERMAL_LOAD = load_bdew_demand_from_csv(WEATHER_YEAR)
    print(f"Thermal load (BDEW): {len(P_THERMAL_LOAD)} timesteps, "
          f"peak = {max(P_THERMAL_LOAD):.2f} kW_th, "
          f"total = {sum(P_THERMAL_LOAD)/1000:.1f} MWh")

    # 2. Calculate the weather-dependent capacity limit using the real temperature
    Cap_frac_t = calculate_capacity_fraction(T_amb)

    # 3. Validation checks
    assert len(P_price_buy) == T, f"Price array length {len(P_price_buy)} != {T}"
    assert len(I_SOLAR)     == T, f"Solar array length {len(I_SOLAR)} != {T}"
    assert len(COP_t)       == T, f"COP array length {len(COP_t)} != {T}"
    assert len(Cap_frac_t)  == T, f"Cap_frac array length {len(Cap_frac_t)} != {T}"
    assert len(P_THERMAL_LOAD) == T, f"Thermal load length {len(P_THERMAL_LOAD)} != {T}"

    # 4. Run the optimizer
    opt, res = run_integrated_optimization(I_SOLAR, COP_t, Cap_frac_t, P_price_buy, P_price_sell)