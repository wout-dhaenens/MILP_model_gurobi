import pulp
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import requests
from fetch_be_prices import fetch_day_ahead_prices_be



# ==============================================================================
# --- 1. GLOBAL PARAMETERS & PHYSICAL CONSTANTS ---
# ==============================================================================

time = np.arange(24)
T = len(time)
dt = 1.0  # hours
BIG_M = 1e6

# --- Electrical System ---
CAPEX_BAT_ANNUAL = 56.17   # €/kWh
CAPEX_PV_ANNUAL  = 80.0    # €/kW_p
C_CAP            = 10.0    # €/kW/month — capacity tariff rate (peak penalty)
ETA_BAT_CH  = 0.95
ETA_BAT_DIS = 0.95
P_BAT_POWER_RATIO = 1.0    # kW per kWh
d       = 1.0    # Tank diameter [m]
# --- Heat Pump & TES ---
CAPEX_HP_ANNUAL        = 150.0   # €/kW_th
CAPEX_TES_eu_L = 3# euro/L
CAPEX_TES_METER_ANNUAL = CAPEX_TES_eu_L * np.pi*d **2/4*1000   # €/m (tank height) €/L*L/m
d       = 1.0    # Tank diameter [m]
rho     = 971.8  # Water density [kg/m³]
c_water = 4190   # Specific heat capacity [J/(kg·K)]
temp_h, temp_c, temp_env = 70, 50, 10
eta_in, eta_out = 0.95, 0.95
capex_TES_annual = CAPEX_TES_METER_ANNUAL/ (rho*np.pi*d **2/4*c_water*20/3600/1000)
print(capex_TES_annual)
delta_T_HC = temp_h - temp_c
delta_T_C0 = temp_c - temp_env
delta_T_H0 = temp_h - temp_env
dt_sec = 3600

# Linear TES coefficients
kWh_per_m    = (np.pi * (d**2 / 4) * rho * c_water * delta_T_HC) / 3.6e6
beta         = (4 / (d * rho * c_water)) * dt_sec
gamma        = (4 / (d * rho * c_water * delta_T_HC)) * delta_T_C0 * dt_sec
loss_lids_kwh = ((2 * np.pi * (d**2 / 4)) * (delta_T_H0 + delta_T_C0) * dt_sec / 2) / 3.6e6

# --- Demand Profiles ---
P_LOAD = np.array([0.5, 0.5, 0.5, 0.6, 0.8, 1.2, 2.5, 3.5, 4.5, 5.0,
                   5.5, 5.8, 6.0, 5.9, 5.5, 5.2, 4.8, 3.5, 1.5, 1.0,
                   0.7, 0.6, 0.5, 0.5])

P_THERMAL_LOAD = 5*np.array([2.0, 2.0, 2.0, 2.5, 3.0, 5.0, 8.0, 10.0, 8.0, 6.0,
                            5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 8.0, 10.0, 9.0, 7.0,
                            5.0, 3.0, 2.0, 2.0])


# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def calculate_lorenz_cop(t_sink_in, t_sink_out, t_source_in):
    T_in  = t_sink_in  + 273.15
    T_out = t_sink_out + 273.15
    T_src = t_source_in + 273.15
    if T_in == T_out:
        T_h_avg = T_in
    else:
        T_h_avg = (T_out - T_in) / math.log(T_out / T_in)
    return T_h_avg / (T_h_avg - T_src)


def get_pvgis_inputs(lat, lon, peak_power=1.0, date_str='2023-01-15', tilt=35, azimuth=-90):
    target_date = pd.to_datetime(date_str)
    year = target_date.year
    url = (f"https://re.jrc.ec.europa.eu/api/seriescalc?lat={lat}&lon={lon}"
           f"&startyear={year}&endyear={year}"
           f"&pvcalculation=1&peakpower={peak_power}"
           f"&angle={tilt}&aspect={azimuth}"
           f"&loss=14&outputformat=json")
    try:
        response = requests.get(url, timeout=15)
        data = response.json()
        df = pd.DataFrame(data['outputs']['hourly'])
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
        day_df = df[df['time'].dt.date == target_date.date()].reset_index(drop=True)
        day_df['cop_hthp'] = day_df['T2m'].apply(
            lambda t: calculate_lorenz_cop(temp_c, temp_h, t))
        day_df['pv_kw'] = day_df['P'] / 1000
        print("PVGIS data fetched successfully.")
        return day_df['pv_kw'].values, day_df['cop_hthp'].values
    except Exception as e:
        print(f"PVGIS fetch failed ({e}). Using synthetic profiles.")
        I_SOLAR = np.array([max(0, 8 * np.sin(np.pi * (t - 6) / 12)) if 6 <= t <= 18 else 0
                            for t in range(24)])
        COP_t = np.full(24, calculate_lorenz_cop(temp_c, temp_h, 10.0))
        return I_SOLAR, COP_t


def load_price_data(filename="load_data.csv"):
    fallback = np.array([0.1]*4 + [0.12, 0.15, 0.4, 0.5, 0.4, 0.2, 0.15] +
                        [0.1]*5 + [0.15, 0.3, 0.5, 0.6, 0.4, 0.2] + [0.1]*3)
    try:
        try:
            df = pd.read_csv(filename, sep=';', encoding='windows-1252')
        except Exception:
            df = pd.read_csv(filename, sep=',', encoding='windows-1252')
        df = df.iloc[:, 0:2]
        df.columns = ['Date', 'Euro']
        cleaned = (df['Euro'].astype(str)
                   .str.replace(r'[^\d,]', '', regex=True)
                   .str.replace(',', '.', regex=False)
                   .astype(float))
        prices = (cleaned.to_numpy() / 1000)[::-1]
        if len(prices) < T:
            prices = np.pad(prices, (0, T - len(prices)), mode='edge')
        return prices[:T]
    except Exception as e:
        print(f"Price data load failed ({e}). Using synthetic prices.")
        return fallback


# ==============================================================================
# --- 3. INTEGRATED MILP ---
# ==============================================================================

def run_integrated_optimization(I_SOLAR, COP_t, P_price):

    prob = pulp.LpProblem("Integrated_MILP", pulp.LpMinimize)

    # --- Design Variables ---
    C_PV   = pulp.LpVariable('C_PV',   lowBound=0)
    C_bat  = pulp.LpVariable('C_bat',  lowBound=0)
    C_HP   = pulp.LpVariable('C_HP',   lowBound=0)
    h_tank = pulp.LpVariable('h_tank', lowBound=0.5, upBound=10.0)

    y_PV  = pulp.LpVariable('y_PV',  cat='Binary')
    y_bat = pulp.LpVariable('y_bat', cat='Binary')
    y_HP  = pulp.LpVariable('y_HP',  cat='Binary')

    # --- Operational Variables ---
    P_buy      = pulp.LpVariable.dicts('P_buy',      time, lowBound=0)
    P_sell     = pulp.LpVariable.dicts('P_sell',     time, lowBound=0)
    P_bat_ch   = pulp.LpVariable.dicts('P_bat_ch',   time, lowBound=0)
    P_bat_dis  = pulp.LpVariable.dicts('P_bat_dis',  time, lowBound=0)
    SoC_bat    = pulp.LpVariable.dicts('SoC_bat',    time, lowBound=0)
    P_hp_elec  = pulp.LpVariable.dicts('P_hp_elec',  time, lowBound=0)
    Q_tes      = pulp.LpVariable.dicts('Q_tes',      time, lowBound=0)

    # --- §1: Peak variable P_peak >= 0 ---
    P_peak = pulp.LpVariable('P_peak', lowBound=0)

    C_TES = h_tank * kWh_per_m

    # --- Objective ---
    inv_cost = (CAPEX_PV_ANNUAL * C_PV
                + CAPEX_BAT_ANNUAL * C_bat
                + CAPEX_HP_ANNUAL * C_HP
                + CAPEX_TES_METER_ANNUAL * h_tank)

    P_feedin = P_price * 0.1
    op_buy  = pulp.lpSum(P_price[t]  * P_buy[t]  * dt for t in time)
    op_sell = pulp.lpSum(P_feedin[t] * P_sell[t] * dt for t in time)

    # --- §3: Objective += C_cap * P_peak (annualised: 12 months) ---
    cap_tariff_cost = 12 * C_CAP * P_peak

    prob += inv_cost + 365 * (op_buy - op_sell) + cap_tariff_cost, "Total_Annual_Cost"

    # --- Constraints ---
    prob += C_PV  <= BIG_M * y_PV,  "Cap_PV"
    prob += C_bat <= BIG_M * y_bat, "Cap_bat"
    prob += C_HP  <= BIG_M * y_HP,  "Cap_HP"

    for t in time:
        prev = T - 1 if t == 0 else t - 1
        q_hp_th_t = P_hp_elec[t] * COP_t[t]
        PV_t = I_SOLAR[t] * C_PV

        # Electrical balance
        prob += (PV_t + P_bat_dis[t] + P_buy[t]
                 == P_LOAD[t] + P_bat_ch[t] + P_sell[t] + P_hp_elec[t],
                 f"Elec_Balance_t{t}")

        # Battery
        prob += P_bat_ch[t]  <= P_BAT_POWER_RATIO * C_bat, f"Bat_Ch_t{t}"
        prob += P_bat_dis[t] <= P_BAT_POWER_RATIO * C_bat, f"Bat_Dis_t{t}"
        prob += SoC_bat[t]   <= C_bat,                      f"SoC_Max_t{t}"
        prob += (SoC_bat[t] == SoC_bat[prev]
                 + (ETA_BAT_CH * P_bat_ch[t] - (1 / ETA_BAT_DIS) * P_bat_dis[t]) * dt,
                 f"SoC_Dyn_t{t}")

        # Heat pump
        prob += q_hp_th_t <= C_HP, f"HP_Cap_t{t}"

        # TES balance
        loss_t = beta * Q_tes[prev] + gamma * C_TES + loss_lids_kwh
        prob += (Q_tes[t] == Q_tes[prev]
                 - loss_t
                 + eta_in  * q_hp_th_t
                 - (1 / eta_out) * P_THERMAL_LOAD[t],
                 f"TES_Dyn_t{t}")
        prob += Q_tes[t] >= 0.05 * C_TES, f"TES_Min_t{t}"
        prob += Q_tes[t] <= 0.95 * C_TES, f"TES_Max_t{t}"

        # --- §2: P_peak >= p_t  ∀ t ∈ T  (p_t = P_buy[t], net grid import) ---
        prob += P_peak >= P_buy[t], f"Peak_Track_t{t}"

    # Periodicity
    prob += SoC_bat[T - 1] == SoC_bat[0], "Bat_Cyclic"
    prob += Q_tes[T - 1]   == Q_tes[0],   "TES_Cyclic"

    # --- Solve ---
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    print(f"\nSolver status: {status}")

    if status != "Optimal":
        print(" No optimal solution found.")
        return None, None

    # --- Extract Results ---
    opt = {
        'C_PV':   pulp.value(C_PV),
        'C_bat':  pulp.value(C_bat),
        'C_HP':   pulp.value(C_HP),
        'h_tank': pulp.value(h_tank),
        'P_peak': pulp.value(P_peak),
        'obj':    pulp.value(prob.objective),
    }
    opt['C_TES'] = opt['h_tank'] * kWh_per_m

    arr = lambda d: np.array([pulp.value(d[t]) for t in time])
    res = {
        'SoC_bat':   arr(SoC_bat),
        'P_buy':     arr(P_buy),
        'P_sell':    arr(P_sell),
        'P_bat_ch':  arr(P_bat_ch),
        'P_bat_dis': arr(P_bat_dis),
        'P_hp_elec': arr(P_hp_elec),
        'Q_tes':     arr(Q_tes),
        'PV_prod':   I_SOLAR * opt['C_PV'],
        'Q_hp_th':   arr(P_hp_elec) * COP_t
    }

    res['TES_loss'] = np.array([
        beta * res['Q_tes'][T - 1 if t == 0 else t - 1]
        + gamma * opt['C_TES'] + loss_lids_kwh
        for t in time
    ])
    volume_liters = opt['h_tank'] * np.pi*d**2/4 * 1000
    print(f"  PV:     {opt['C_PV']:.2f} kWp")
    print(f"  Bat:    {opt['C_bat']:.2f} kWh")
    print(f"  HP:     {opt['C_HP']:.2f} kW_th")
    print(f"  TES:    h={opt['h_tank']:.2f} m  |  V={volume_liters:.1f} L  →  {opt['C_TES']:.1f} kWh")
    print(f"  P_peak: {opt['P_peak']:.2f} kW  (capacity tariff penalty: €{12 * C_CAP * opt['P_peak']:,.0f}/yr)")
    print(f"  Annualised total cost: €{opt['obj']:,.0f}")

    _plot_results(opt, res, P_price, COP_t)
    return opt, res


# ==============================================================================
# --- 4. PLOTTING ---
# ==============================================================================

def _plot_results(opt, res, P_price, COP_t):
    fig, axes = plt.subplots(4, 1, figsize=(13, 18), sharex=True)

    title = (f"Integrated Optimisation Results\n"
             f"PV {opt['C_PV']:.1f} kWp | Bat {opt['C_bat']:.1f} kWh | "
             f"HP {opt['C_HP']:.1f} kW_th | TES h={opt['h_tank']:.2f} m "
             f"({opt['C_TES']:.1f} kWh) | P_peak={opt['P_peak']:.2f} kW")
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.99)

    # Panel 1: Electrical power flows
    ax = axes[0]
    ax.plot(time, res['PV_prod'],   label='PV Generation',  color='orange', lw=2)
    ax.plot(time, P_LOAD,           label='Electrical Load', color='black',  lw=2, ls='--')
    ax.plot(time, res['P_hp_elec'], label='HP Elec. Input',  color='red',    lw=2)
    P_bat_net  = res['P_bat_dis'] - res['P_bat_ch']
    ax.plot(time, P_bat_net,        label='Battery Net',     color='green',  lw=2)
    P_grid_net = res['P_buy'] - res['P_sell']
    ax.fill_between(time, P_grid_net, 0, where=P_grid_net > 0,
                    alpha=0.25, color='steelblue', label='Grid Buy')
    ax.fill_between(time, P_grid_net, 0, where=P_grid_net < 0,
                    alpha=0.25, color='gold',      label='Grid Sell')
    # --- Peak line ---
    ax.axhline(opt['P_peak'], color='red', ls='--', lw=1.5,
               label=f"P_peak = {opt['P_peak']:.2f} kW")
    ax.set_ylabel('Power [kW_e]', fontsize=12)
    ax.legend(fontsize=10, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Electrical Power Flows', fontsize=11)

    # Panel 2: Thermal system
    ax = axes[1]
    ax.plot(time, P_THERMAL_LOAD,    label='Thermal Demand',       color='black',      lw=2, ls='--')
    ax.plot(time, res['Q_hp_th'],    label='HP Output [kW_th]',    color='red',        lw=2)
    ax2 = ax.twinx()
    ax2.plot(time, res['Q_tes'], label='TES SoC [kWh]', color='blue', lw=3)
    ax2.axhline(0.95 * opt['C_TES'], color='blue', ls=':', alpha=0.6, label='TES 95%')
    ax2.axhline(0.05 * opt['C_TES'], color='blue', ls=':', alpha=0.4)
    ax2.set_ylabel('Stored Energy [kWh]', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax.set_ylabel('Power [kW_th]', fontsize=12)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Thermal System (HP & TES)', fontsize=11)

    # Panel 3: Battery SoC & COP
    ax = axes[2]
    ax.plot(time, res['SoC_bat'], label='Battery SoC [kWh]', color='green', lw=3)
    ax.axhline(opt['C_bat'], color='green', ls='--', alpha=0.5, label='Bat Capacity')
    ax3 = ax.twinx()
    ax3.plot(time, COP_t, label='Lorenz COP', color='purple', lw=2, ls='-.')
    ax3.set_ylabel('COP [-]', color='purple', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='purple')
    ax.set_ylabel('Energy [kWh]', fontsize=12)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Battery SoC & Heat Pump COP', fontsize=11)

    # Panel 4: Prices & TES losses
    ax = axes[3]
    ax.step(time, P_price,       label='Buy Price [€/kWh]',  color='purple', lw=2,   where='post')
    ax.step(time, P_price * 0.1, label='Sell Price [€/kWh]', color='gray',   lw=1.5, ls='--', where='post')
    ax4 = ax.twinx()
    ax4.bar(time, res['TES_loss'], alpha=0.4, color='darkred', label='TES Losses [kWh]')
    ax4.set_ylabel('TES Losses [kWh]', color='darkred', fontsize=12)
    ax4.tick_params(axis='y', labelcolor='darkred')
    ax.set_ylabel('Price [€/kWh]', fontsize=12)
    ax.set_xlabel('Hour of Day', fontsize=12)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax4.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Electricity Prices & TES Thermal Losses', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# ==============================================================================
# --- 5. MAIN EXECUTION ---
# ==============================================================================

if __name__ == "__main__":
    DATE = "2023-04-15"   # ← change to any date you want

    P_price = fetch_day_ahead_prices_be(DATE)
    I_SOLAR, COP_t = get_pvgis_inputs(lat=50.85, lon=4.35, date_str=DATE)

    opt, res = run_integrated_optimization(I_SOLAR, COP_t, P_price)