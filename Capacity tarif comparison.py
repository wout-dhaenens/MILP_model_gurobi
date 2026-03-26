import pulp
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests

# Try to import real price fetcher, fall back to synthetic
try:
    from fetch_be_prices import fetch_day_ahead_prices_be
    USE_REAL_PRICES = True
except ImportError:
    USE_REAL_PRICES = False


# ==============================================================================
# --- 1. GLOBAL PARAMETERS & PHYSICAL CONSTANTS ---
# ==============================================================================

time = np.arange(24)
T = len(time)
dt = 1.0
BIG_M = 1e6

# --- Electrical System ---
CAPEX_BAT_ANNUAL  = 56.17
CAPEX_PV_ANNUAL   = 80.0
ETA_BAT_CH        = 0.95
ETA_BAT_DIS       = 0.95
P_BAT_POWER_RATIO = 1.0

# --- Heat Pump & TES ---
CAPEX_HP_ANNUAL        = 150.0
d                      = 1.0
CAPEX_TES_eu_L         = 3
CAPEX_TES_METER_ANNUAL = CAPEX_TES_eu_L * np.pi * d**2 / 4 * 1000
rho     = 971.8
c_water = 4190
temp_h, temp_c, temp_env = 70, 50, 10
eta_in, eta_out = 0.95, 0.95
delta_T_HC = temp_h - temp_c
delta_T_C0 = temp_c - temp_env
delta_T_H0 = temp_h - temp_env
dt_sec = 3600

kWh_per_m    = (np.pi * (d**2 / 4) * rho * c_water * delta_T_HC) / 3.6e6
beta         = (4 / (d * rho * c_water)) * dt_sec
gamma        = (4 / (d * rho * c_water * delta_T_HC)) * delta_T_C0 * dt_sec
loss_lids_kwh = ((2 * np.pi * (d**2 / 4)) * (delta_T_H0 + delta_T_C0) * dt_sec / 2) / 3.6e6

# --- Demand Profiles ---
P_LOAD = np.array([0.5, 0.5, 0.5, 0.6, 0.8, 1.2, 2.5, 3.5, 4.5, 5.0,
                   5.5, 5.8, 6.0, 5.9, 5.5, 5.2, 4.8, 3.5, 1.5, 1.0,
                   0.7, 0.6, 0.5, 0.5])

P_THERMAL_LOAD = 5 * np.array([2.0, 2.0, 2.0, 2.5, 3.0, 5.0, 8.0, 10.0, 8.0, 6.0,
                                5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 8.0, 10.0, 9.0, 7.0,
                                5.0, 3.0, 2.0, 2.0])

# --- Capacity tariff rates to sweep ---
C_CAP_VALUES = [0, 5, 10, 20, 40]   # €/kW/month  (0 = no peak penalty)
DATE = "2023-04-15"


# ==============================================================================
# --- 2. HELPERS ---
# ==============================================================================

def calculate_lorenz_cop(t_sink_in, t_sink_out, t_source_in):
    T_in  = t_sink_in  + 273.15
    T_out = t_sink_out + 273.15
    T_src = t_source_in + 273.15
    T_h_avg = (T_out - T_in) / math.log(T_out / T_in) if T_in != T_out else T_in
    return T_h_avg / (T_h_avg - T_src)


def get_pvgis_inputs(lat, lon, date_str='2023-04-15', tilt=35, azimuth=-90):
    target_date = pd.to_datetime(date_str)
    year = target_date.year
    url = (f"https://re.jrc.ec.europa.eu/api/seriescalc?lat={lat}&lon={lon}"
           f"&startyear={year}&endyear={year}"
           f"&pvcalculation=1&peakpower=1.0"
           f"&angle={tilt}&aspect={azimuth}"
           f"&loss=14&outputformat=json")
    try:
        response = requests.get(url, timeout=15)
        data = response.json()
        df = pd.DataFrame(data['outputs']['hourly'])
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
        day_df = df[df['time'].dt.date == target_date.date()].reset_index(drop=True)
        day_df['cop'] = day_df['T2m'].apply(lambda t: calculate_lorenz_cop(temp_c, temp_h, t))
        return day_df['P'].values / 1000, day_df['cop'].values
    except Exception as e:
        print(f"PVGIS fetch failed ({e}). Using synthetic profiles.")
        I_SOLAR = np.array([max(0, 8 * np.sin(np.pi * (t - 6) / 12)) if 6 <= t <= 18 else 0
                            for t in range(24)])
        COP_t = np.full(24, calculate_lorenz_cop(temp_c, temp_h, 10.0))
        return I_SOLAR, COP_t


def get_prices(date_str):
    if USE_REAL_PRICES:
        try:
            return fetch_day_ahead_prices_be(date_str)
        except Exception:
            pass
    # Synthetic fallback
    return np.array([0.10]*4 + [0.12, 0.15, 0.40, 0.50, 0.40, 0.20, 0.15]
                    + [0.10]*5 + [0.15, 0.30, 0.50, 0.60, 0.40, 0.20] + [0.10]*3)


# ==============================================================================
# --- 3. MILP SOLVER (with optional peak penalty) ---
# ==============================================================================

def run_milp(I_SOLAR, COP_t, P_price, C_cap=0.0, label=""):
    """
    Runs the integrated MILP.
    C_cap = 0  →  no capacity tariff (baseline)
    C_cap > 0  →  capacity tariff active at €/kW/month
    """
    prob = pulp.LpProblem(f"MILP_{label}", pulp.LpMinimize)

    # Design variables
    C_PV   = pulp.LpVariable('C_PV',   lowBound=0)
    C_bat  = pulp.LpVariable('C_bat',  lowBound=0)
    C_HP   = pulp.LpVariable('C_HP',   lowBound=0)
    h_tank = pulp.LpVariable('h_tank', lowBound=0.5, upBound=10.0)
    y_PV   = pulp.LpVariable('y_PV',  cat='Binary')
    y_bat  = pulp.LpVariable('y_bat', cat='Binary')
    y_HP   = pulp.LpVariable('y_HP',  cat='Binary')

    # Operational variables
    P_buy     = pulp.LpVariable.dicts('P_buy',     time, lowBound=0)
    P_sell    = pulp.LpVariable.dicts('P_sell',    time, lowBound=0)
    P_bat_ch  = pulp.LpVariable.dicts('P_bat_ch',  time, lowBound=0)
    P_bat_dis = pulp.LpVariable.dicts('P_bat_dis', time, lowBound=0)
    SoC_bat   = pulp.LpVariable.dicts('SoC_bat',   time, lowBound=0)
    P_hp_elec = pulp.LpVariable.dicts('P_hp_elec', time, lowBound=0)
    Q_tes     = pulp.LpVariable.dicts('Q_tes',     time, lowBound=0)

    # §1 – Peak variable
    P_peak = pulp.LpVariable('P_peak', lowBound=0)

    C_TES = h_tank * kWh_per_m

    # Objective
    inv_cost = (CAPEX_PV_ANNUAL * C_PV + CAPEX_BAT_ANNUAL * C_bat
                + CAPEX_HP_ANNUAL * C_HP + CAPEX_TES_METER_ANNUAL * h_tank)
    P_feedin = P_price * 0.1
    op_buy   = pulp.lpSum(P_price[t] * P_buy[t]  * dt for t in time)
    op_sell  = pulp.lpSum(P_feedin[t] * P_sell[t] * dt for t in time)
    cap_cost = 12 * C_cap * P_peak   # annualised

    prob += inv_cost + 365 * (op_buy - op_sell) + cap_cost, "Total_Annual_Cost"

    # Capacity constraints
    prob += C_PV  <= BIG_M * y_PV
    prob += C_bat <= BIG_M * y_bat
    prob += C_HP  <= BIG_M * y_HP

    for t in time:
        prev = T - 1 if t == 0 else t - 1
        q_hp_th_t = P_hp_elec[t] * COP_t[t]
        PV_t = I_SOLAR[t] * C_PV

        # Electrical balance
        prob += (PV_t + P_bat_dis[t] + P_buy[t]
                 == P_LOAD[t] + P_bat_ch[t] + P_sell[t] + P_hp_elec[t],
                 f"Elec_{t}")

        # Battery
        prob += P_bat_ch[t]  <= P_BAT_POWER_RATIO * C_bat
        prob += P_bat_dis[t] <= P_BAT_POWER_RATIO * C_bat
        prob += SoC_bat[t]   <= C_bat
        prob += (SoC_bat[t] == SoC_bat[prev]
                 + (ETA_BAT_CH * P_bat_ch[t] - (1/ETA_BAT_DIS) * P_bat_dis[t]) * dt)

        # Heat pump
        prob += q_hp_th_t <= C_HP

        # TES
        loss_t = beta * Q_tes[prev] + gamma * C_TES + loss_lids_kwh
        prob += Q_tes[t] == Q_tes[prev] - loss_t + eta_in * q_hp_th_t - (1/eta_out) * P_THERMAL_LOAD[t]
        prob += Q_tes[t] >= 0.05 * C_TES
        prob += Q_tes[t] <= 0.95 * C_TES

        # §2 – Peak tracking
        prob += P_peak >= P_buy[t], f"Peak_{t}"

    # Periodicity
    prob += SoC_bat[T-1] == SoC_bat[0]
    prob += Q_tes[T-1]   == Q_tes[0]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        print(f"  [{label}] No optimal solution: {status}")
        return None

    arr = lambda d: np.array([pulp.value(d[t]) for t in time])
    C_PV_val   = pulp.value(C_PV)
    C_bat_val  = pulp.value(C_bat)
    C_HP_val   = pulp.value(C_HP)
    h_tank_val = pulp.value(h_tank)
    C_TES_val  = h_tank_val * kWh_per_m
    P_peak_val = pulp.value(P_peak)
    obj_val    = pulp.value(prob.objective)

    P_buy_arr  = arr(P_buy)
    P_sell_arr = arr(P_sell)

    energy_cost_val = float(365 * np.sum(P_price * P_buy_arr * dt)
                            - 365 * np.sum(P_price * 0.1 * P_sell_arr * dt))
    inv_cost_val    = float(pulp.value(inv_cost))
    cap_cost_val    = float(12 * C_cap * P_peak_val)

    return {
        'label':       label,
        'C_cap':       C_cap,
        'C_PV':        C_PV_val,
        'C_bat':       C_bat_val,
        'C_HP':        C_HP_val,
        'h_tank':      h_tank_val,
        'C_TES':       C_TES_val,
        'P_peak':      P_peak_val,
        'obj':         obj_val,
        'energy_cost': energy_cost_val,
        'inv_cost':    inv_cost_val,
        'cap_cost':    cap_cost_val,
        'P_buy':       P_buy_arr,
        'P_sell':      P_sell_arr,
        'P_bat_ch':    arr(P_bat_ch),
        'P_bat_dis':   arr(P_bat_dis),
        'SoC_bat':     arr(SoC_bat),
        'P_hp_elec':   arr(P_hp_elec),
        'Q_tes':       arr(Q_tes),
        'PV_prod':     I_SOLAR * C_PV_val,
        'Q_hp_th':     arr(P_hp_elec) * COP_t,
    }


# ==============================================================================
# --- 4. COMPARISON PLOTS ---
# ==============================================================================

COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']


def plot_comparison(results, P_price):
    """
    Figure 1 – Side-by-side grid import profiles for baseline vs peak cases.
    Figure 2 – Sensitivity: how key metrics evolve with C_cap.
    Figure 3 – Cost breakdown comparison.
    """

    baseline = results[0]   # C_cap = 0
    with_peak = results[1]  # C_cap = first non-zero

    # ------------------------------------------------------------------
    # Figure 1: Head-to-head operational comparison (baseline vs C_cap=10)
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True)
    fig1.suptitle(
        f"Operational Comparison: No Peak Penalty vs C_cap={with_peak['C_cap']} €/kW/month",
        fontsize=15, fontweight='bold'
    )

    case_pair = [baseline, with_peak]
    col_titles = [f"Baseline  (C_cap = 0 €/kW/month)\nP_peak = {baseline['P_peak']:.2f} kW",
                  f"With Peak Penalty  (C_cap = {with_peak['C_cap']} €/kW/month)\nP_peak = {with_peak['P_peak']:.2f} kW"]

    for col, res in enumerate(case_pair):
        P_grid_net = res['P_buy'] - res['P_sell']
        P_bat_net  = res['P_bat_dis'] - res['P_bat_ch']

        # Row 0: Electrical flows
        ax = axes[0, col]
        ax.plot(time, res['PV_prod'],   color='orange', lw=2, label='PV')
        ax.plot(time, P_LOAD,           color='black',  lw=2, ls='--', label='Elec Load')
        ax.plot(time, res['P_hp_elec'], color='red',    lw=2, label='HP elec')
        ax.plot(time, P_bat_net,        color='green',  lw=2, label='Bat net')
        ax.fill_between(time, P_grid_net, 0, where=P_grid_net > 0,
                        alpha=0.3, color='steelblue', label='Grid buy')
        ax.fill_between(time, P_grid_net, 0, where=P_grid_net < 0,
                        alpha=0.3, color='gold', label='Grid sell')
        ax.axhline(res['P_peak'], color='red', ls='--', lw=1.5,
                   label=f"P_peak={res['P_peak']:.1f} kW")
        ax.set_title(col_titles[col], fontsize=11, fontweight='bold')
        ax.set_ylabel('Power [kW_e]')
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3)

        # Row 1: Battery SoC
        ax = axes[1, col]
        ax.plot(time, res['SoC_bat'], color='green', lw=2.5, label='SoC [kWh]')
        ax.axhline(res['C_bat'], color='green', ls=':', alpha=0.6,
                   label=f"Capacity={res['C_bat']:.1f} kWh")
        ax.set_ylabel('Battery SoC [kWh]')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Row 2: TES SoC
        ax = axes[2, col]
        ax.plot(time, res['Q_tes'], color='blue', lw=2.5, label='TES SoC [kWh]')
        ax.axhline(0.95 * res['C_TES'], color='blue', ls=':', alpha=0.5,
                   label=f"95% = {0.95*res['C_TES']:.1f} kWh")
        ax.set_ylabel('TES SoC [kWh]')
        ax.set_xlabel('Hour of Day')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_operational.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ------------------------------------------------------------------
    # Figure 2: Sensitivity to C_cap
    # ------------------------------------------------------------------
    caps       = [r['C_cap']   for r in results]
    peaks      = [r['P_peak']  for r in results]
    c_pvs      = [r['C_PV']    for r in results]
    c_bats     = [r['C_bat']   for r in results]
    c_hps      = [r['C_HP']    for r in results]
    h_tanks    = [r['h_tank']  for r in results]
    objs       = [r['obj']     for r in results]
    en_costs   = [r['energy_cost'] for r in results]
    cap_costs  = [r['cap_cost']    for r in results]
    inv_costs  = [r['inv_cost']    for r in results]

    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9))
    fig2.suptitle("Sensitivity Analysis: Effect of Capacity Tariff Rate (C_cap)",
                  fontsize=14, fontweight='bold')

    def bar_ax(ax, y, title, ylabel, color):
        bars = ax.bar(caps, y, color=color, alpha=0.85, edgecolor='white', width=2.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('C_cap [€/kW/month]')
        ax.set_ylabel(ylabel)
        ax.set_xticks(caps)
        for bar, val in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    bar_ax(axes2[0, 0], peaks,   'Peak Grid Import (P_peak)', 'kW',    '#E91E63')
    bar_ax(axes2[0, 1], c_pvs,   'Optimal PV Capacity',       'kWp',   '#FF9800')
    bar_ax(axes2[0, 2], c_bats,  'Optimal Battery Capacity',  'kWh',   '#4CAF50')
    bar_ax(axes2[1, 0], c_hps,   'Optimal HP Capacity',       'kW_th', '#2196F3')
    bar_ax(axes2[1, 1], h_tanks, 'Optimal TES Height',        'm',     '#9C27B0')
    bar_ax(axes2[1, 2], objs,    'Total Annualised Cost',      '€/yr',  '#607D8B')

    plt.tight_layout()
    plt.savefig('comparison_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ------------------------------------------------------------------
    # Figure 3: Stacked cost breakdown
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    x   = np.arange(len(caps))
    w   = 0.5

    bars_inv = ax3.bar(x, inv_costs,  w, label='Investment Cost [€/yr]', color='#607D8B')
    bars_en  = ax3.bar(x, en_costs,   w, bottom=inv_costs,
                       label='Net Energy Cost [€/yr]', color='#2196F3')
    bottom2  = [i + e for i, e in zip(inv_costs, en_costs)]
    bars_cap = ax3.bar(x, cap_costs,  w, bottom=bottom2,
                       label='Capacity Tariff Cost [€/yr]', color='#E91E63', alpha=0.85)

    ax3.set_xticks(x)
    ax3.set_xticklabels([f"C_cap={c}\n€/kW/mo" for c in caps])
    ax3.set_ylabel('Annual Cost [€]', fontsize=12)
    ax3.set_title('Annual Cost Breakdown by Capacity Tariff Rate', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)

    # Total cost labels on top
    for xi, total in zip(x, objs):
        ax3.text(xi, total + max(objs)*0.01, f'€{total:,.0f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('comparison_cost_breakdown.png', dpi=150, bbox_inches='tight')
    plt.show()


# ==============================================================================
# --- 5. PRINT SUMMARY TABLE ---
# ==============================================================================

def print_summary(results):
    print("\n" + "="*90)
    print(f"{'C_cap':>8} | {'P_peak':>8} | {'C_PV':>7} | {'C_bat':>7} | "
          f"{'C_HP':>7} | {'h_TES':>7} | {'Inv €':>9} | {'Energy €':>10} | {'Cap €':>8} | {'Total €':>9}")
    print(f"{'€/kW/mo':>8} | {'kW':>8} | {'kWp':>7} | {'kWh':>7} | "
          f"{'kW_th':>7} | {'m':>7} | {'':>9} | {'':>10} | {'':>8} | {'':>9}")
    print("-"*90)
    for r in results:
        print(f"{r['C_cap']:>8.0f} | {r['P_peak']:>8.2f} | {r['C_PV']:>7.2f} | "
              f"{r['C_bat']:>7.2f} | {r['C_HP']:>7.2f} | {r['h_tank']:>7.2f} | "
              f"{r['inv_cost']:>9,.0f} | {r['energy_cost']:>10,.0f} | "
              f"{r['cap_cost']:>8,.0f} | {r['obj']:>9,.0f}")
    print("="*90)

    # Delta vs baseline
    baseline = results[0]
    print("\n--- Delta vs Baseline (C_cap=0) ---")
    for r in results[1:]:
        dpeak = r['P_peak'] - baseline['P_peak']
        dobj  = r['obj']    - baseline['obj']
        print(f"  C_cap={r['C_cap']:>4} €/kW/mo  →  "
              f"ΔP_peak={dpeak:+.2f} kW  |  ΔTotal cost={dobj:+,.0f} €/yr")


# ==============================================================================
# --- 6. MAIN ---
# ==============================================================================

if __name__ == "__main__":
    print(f"Fetching data for {DATE} ...")
    P_price = get_prices(DATE)
    I_SOLAR, COP_t = get_pvgis_inputs(lat=50.85, lon=4.35, date_str=DATE)

    print(f"\nRunning {len(C_CAP_VALUES)} MILP scenarios ...\n")
    results = []
    for c_cap in C_CAP_VALUES:
        label = f"C_cap={c_cap}"
        print(f"  Solving: {label} €/kW/month ...")
        res = run_milp(I_SOLAR, COP_t, P_price, C_cap=c_cap, label=label)
        if res:
            results.append(res)
            print(f"    → P_peak={res['P_peak']:.2f} kW  |  Total={res['obj']:,.0f} €/yr")

    print_summary(results)
    plot_comparison(results, P_price)