"""
baseline_reference_cost.py
==========================
Calculates the total annualised cost of KEEPING the existing installation
as seen from the OUTSIDER VIEWPOINT PRINCIPLE:

  "What would a rational outside investor have to pay, starting from today,
   to deliver the same energy services using the same technology?"

Sunk costs are irrelevant.  CAPEX is annualised over the REMAINING useful
life of each asset (since an outsider committing to this system only faces
the future replacement cycle, not the past one).

Existing installation
---------------------
  - 2 × 80 kW condensing gas boilers  (160 kW_th total)
  - 35 kWp PV array (east-facing, tilt 35°)
  - 1 000 L sensible-water TES buffer
  - Age: 8 years at the time of comparison
"""

import numpy as np
import pandas as pd
import json
import math

from Fetch_and_save_data import (load_pvgis_from_csv, load_prices_from_csv,
                                  load_prices_from_epex, fetch_pvgis_to_csv)

# ==============================================================================
# --- 1. PARAMETERS & CONSTANTS  (mirror Milp_yearly_test_gurobi.py where relevant)
# ==============================================================================

YEAR          = 2023
T             = 8760          # hours in a year
dt            = 1.0           # h
VAT           = 0.21
DISCOUNT_RATE = 0.05

PRICE_SOURCE  = "csv"         # "csv" or "epex"
EPEX_CSV_PATH = "epex_2025.csv"

# --- CRF helper ---
def crf(r, n):
    """Capital Recovery Factor — annualises a one-time CAPEX over n years."""
    if r == 0:
        return 1.0 / n
    return r * (1 + r)**n / ((1 + r)**n - 1)

# ==============================================================================
# --- 2. EXISTING INSTALLATION SPECS
# ==============================================================================

# --- Gas boilers ---
N_BOILERS               = 2
BOILER_UNIT_KW          = 80.0           # kW_th each
BOILER_TOTAL_KW         = N_BOILERS * BOILER_UNIT_KW   # 160 kW_th
ETA_BOILER              = 1           # condensing efficiency [-]
LIFETIME_GAS_BOILER     = 20             # yr (typical for commercial condensing)
AGE_INSTALLATION        = 8              # yr old at time of comparison
CAPEX_BOILER_UNIT       = 3600/30          # €/kW_th installed (commercial condensing, installed)
OPEX_BOILER_MAINT_FRAC  = 0.015         # annual maintenance as fraction of initial CAPEX

# --- Gas price (commodity + network + taxes, excl. VAT) ---
# Source: TotalEnergies proEssential Variabel, tariefkaart maart 2026
# Zone: Fluvius Imewo, consumption tier 5 001–150 000 kWh/yr
#   Energy (TTF-based, current):            3.99 c€/kWh
#   Distribution variable (5001-150 000):   0.97 c€/kWh
#   Transport (Fluxys):                     0.16 c€/kWh
#   Bijdrage op de energie:                 0.10 c€/kWh
#   Federale bijdrage:                      0.07 c€/kWh
#   ─────────────────────────────────────────────────────
#   Total variable:                         5.29 c€/kWh
GAS_PRICE_EUR_KWH       = 0.0529         # €/kWh_gas (excl. VAT)

# Fixed annual gas costs (excl. VAT) — Fluvius Imewo, 5001–150 000 kWh tier
#   Vaste vergoeding leverancier:           35.00 €/yr
#   Distribution fixed (5001–150 000 kWh): 93.22 €/yr
#   Metering / telactiviteit:              17.85 €/yr
FIXED_GAS_ANNUAL        = 146.07         # €/yr (excl. VAT)

# --- EU ETS2 carbon pricing on gas ---
# ETS2 covers buildings & road transport from 2027 onward.
# Cost is passed through by fuel suppliers and is subject to VAT.
# Emission factor for natural gas (LHV basis, IPCC/EEA default):
#   56.1 g CO2/MJ × (1 MJ / 0.2778 kWh) ≈ 0.202 kg CO2/kWh_gas
ETS2_ENABLED            = True           # set False to exclude ETS2
ETS2_PRICE_EUR_TONNE    = 60          # €/tonne CO2 (2027 price cap; expected to rise)
CO2_FACTOR_KG_KWH_GAS   = 0.202         # kg CO2 per kWh_gas (LHV, natural gas)

# --- PV ---
PV_INSTALLED_KWP        = 35.0           # kWp
LIFETIME_PV             = 25             # yr
CAPEX_PV_UNIT           = 900.0          # €/kWp  (same assumption as MILP model)
OPEX_PV_MAINT_FRAC      = 0.01          # 1 %/yr of initial CAPEX

# --- TES (sensible water buffer) ---
TES_INSTALLED_LITRES    = 1000.0         # L
LIFETIME_TES            = 20             # yr
CAPEX_TES_EUR_LITRE     = 3.0            # €/L installed (from MILP model reference)
OPEX_TES_MAINT_FRAC     = 0.005         # 0.5 %/yr

# --- Electrical grid tariffs (same as MILP model) ---
C_CAP                   = 5.0            # €/kW/month  (capacity tariff)
BUY_MARKUP              = 0.083          # €/kWh added on spot
SELL_MARKUP             = 0.03           # €/kWh added on spot for injection
FIXED_ELEC_ANNUAL       = 200.0          # €/yr connection/metering

# ==============================================================================
# --- 3. DEMAND & SOLAR DATA  (reuse loaders from the MILP project)
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

    target_index = pd.date_range(f"{demand_year}-01-01", periods=T, freq='h')
    sub = sub.reindex(target_index)
    if sub.isna().sum() > 0:
        sub = sub.ffill().bfill()

    print(f"Gas CSV loaded: {len(sub)} hourly records for {demand_year}")
    return sub.to_numpy(dtype=float)


METER_CSV_PATH   = "combined_grid_data.csv"
ELEC_DEMAND_YEAR = 2025

def load_electrical_demand_from_meter(csv_path=METER_CSV_PATH, demand_year=ELEC_DEMAND_YEAR):
    df = pd.read_csv(csv_path, sep=";", decimal=",",
                     parse_dates=["timestamp"], index_col="timestamp")
    df = df[df.index.year == demand_year]["Totaal_verbruik_kWh"]
    hourly = df.resample("h").sum()
    target_index = pd.date_range(f"{demand_year}-01-01", periods=T, freq="h")
    hourly = hourly.reindex(target_index)
    if hourly.isna().sum() > 0:
        hourly = hourly.interpolate(method="time").ffill().bfill()
    print(f"Meter CSV loaded: {len(hourly)} hourly records for {demand_year} "
          f"| Total = {hourly.sum():.1f} kWh")
    return hourly.to_numpy(dtype=float)


# ==============================================================================
# --- 4. CAPEX ANNUALISATION (outsider viewpoint → remaining lifetime)
# ==============================================================================

def remaining_life(total_lifetime, age):
    """Years of useful life still ahead of an asset that is <age> years old."""
    rem = total_lifetime - age
    if rem <= 0:
        # Asset is at or past end of life → outsider would need to replace it now
        # Use full new lifetime (immediate replacement cycle)
        return total_lifetime
    return rem


def annualised_capex(capex_total_eur, total_lifetime, age, r=DISCOUNT_RATE):
    """
    Annualised CAPEX under the outsider viewpoint.

    Two components:
      1. Depreciation of RESIDUAL VALUE  (equipment not yet fully depreciated)
         = straight-line remaining book value × CRF(r, remaining_life)
      2. Future replacement at end of remaining life, annualised back to today.

    In practice, for a clean comparison, we use:
      annualised cost = capex_total × CRF(r, remaining_life)
    This asks: "how much per year would an investor pay to finance this asset
    over its remaining horizon?"  It naturally increases as remaining life
    shortens (older assets cost more per year to justify economically).
    """
    rem = remaining_life(total_lifetime, age)
    return capex_total_eur * crf(r, rem)


# ==============================================================================
# --- 5. SIMULATION OF THE OLD INSTALLATION
# ==============================================================================

def simulate_old_installation(I_SOLAR, P_THERMAL_LOAD, P_LOAD, P_price_buy, P_price_sell):
    """
    Simulate the existing installation hour by hour.

    Heat supply:   gas boilers cover 100 % of thermal demand.
    Electricity:   PV generation offsets building load; surplus exported to grid.
    TES:           The 1 000 L buffer is small (~11.6 kWh_th); modelling it as a
                   simple pass-through (no dispatch optimisation) is consistent
                   with the outsider viewpoint — we assume no smart control.

    Returns
    -------
    dict with hourly arrays and scalar summary totals.
    """

    # --- Hourly PV generation [kW] ---
    PV_gen = I_SOLAR * PV_INSTALLED_KWP          # kW

    # --- Electricity balance ---
    # Net load seen by grid = building load - PV (no battery, no HP)
    net_elec = P_LOAD - PV_gen                    # kW  (positive = buy, negative = sell)
    P_buy  = np.maximum(net_elec,  0.0)           # kW
    P_sell = np.maximum(-net_elec, 0.0)           # kW

    # --- Capacity tariff: monthly peak of grid offtake ---
    dates       = pd.date_range(f"{YEAR}-01-01", periods=T, freq='h')
    months      = dates.month.values - 1          # 0-indexed
    P_peak_m    = np.array([
        P_buy[months == m].max() if (months == m).any() else 0.0
        for m in range(12)
    ])

    # --- Gas consumption [kWh_gas] ---
    gas_hourly = P_THERMAL_LOAD / ETA_BOILER      # kWh_gas per hour

    return {
        'PV_gen'   : PV_gen,
        'P_buy'    : P_buy,
        'P_sell'   : P_sell,
        'P_peak_m' : P_peak_m,
        'gas_hourly': gas_hourly,
    }


# ==============================================================================
# --- 6. COST CALCULATION
# ==============================================================================

def calculate_reference_costs(sim, P_price_buy, P_price_sell):
    """
    Break down total annual cost of the existing installation.

    CAPEX   — annualised over remaining useful life (outsider viewpoint).
    OPEX    — actual annual running costs.
    All monetary values include VAT where applicable.
    """

    # ------------------------------------------------------------------
    # CAPEX (annualised, outsider viewpoint, NO VAT on investment)
    # ------------------------------------------------------------------
    capex_boiler_total = CAPEX_BOILER_UNIT  * BOILER_TOTAL_KW         # €
    capex_pv_total     = CAPEX_PV_UNIT      * PV_INSTALLED_KWP        # €
    capex_tes_total    = CAPEX_TES_EUR_LITRE * TES_INSTALLED_LITRES   # €

    capex_boiler_ann   = annualised_capex(capex_boiler_total, LIFETIME_GAS_BOILER, AGE_INSTALLATION)
    capex_pv_ann       = annualised_capex(capex_pv_total,     LIFETIME_PV,         AGE_INSTALLATION)
    capex_tes_ann      = annualised_capex(capex_tes_total,    LIFETIME_TES,        AGE_INSTALLATION)

    capex_total_ann    = capex_boiler_ann + capex_pv_ann + capex_tes_ann

    # ------------------------------------------------------------------
    # OPEX — maintenance (fraction of original CAPEX, excl. VAT base)
    # ------------------------------------------------------------------
    maint_boiler = OPEX_BOILER_MAINT_FRAC * capex_boiler_total   # €/yr excl. VAT
    maint_pv     = OPEX_PV_MAINT_FRAC     * capex_pv_total       # €/yr excl. VAT
    maint_tes    = OPEX_TES_MAINT_FRAC    * capex_tes_total       # €/yr excl. VAT
    maint_total  = (maint_boiler + maint_pv + maint_tes) * (1 + VAT)

    # ------------------------------------------------------------------
    # OPEX — gas (incl. VAT + ETS2 carbon cost)
    # ------------------------------------------------------------------
    gas_annual_kwh   = sim['gas_hourly'].sum()                    # kWh_gas/yr
    gas_cost_excl    = gas_annual_kwh * GAS_PRICE_EUR_KWH + FIXED_GAS_ANNUAL  # €/yr excl. VAT

    # ETS2: fuel suppliers pass through the carbon cost to end users.
    # Calculated on actual gas consumed (not thermal output).
    co2_annual_tonne    = gas_annual_kwh * CO2_FACTOR_KG_KWH_GAS / 1000.0  # tonne CO2/yr
    ets2_cost_excl      = co2_annual_tonne * ETS2_PRICE_EUR_TONNE if ETS2_ENABLED else 0.0
    ets2_cost_incl      = ets2_cost_excl * (1 + VAT)              # €/yr incl. VAT

    gas_cost_incl    = (gas_cost_excl + ets2_cost_excl) * (1 + VAT)  # €/yr incl. VAT

    # ------------------------------------------------------------------
    # OPEX — electricity (incl. VAT)
    # ------------------------------------------------------------------
    opex_buy_excl    = float(np.sum(P_price_buy  * sim['P_buy']  * dt))
    opex_sell_excl   = float(np.sum(P_price_sell * sim['P_sell'] * dt))
    cap_tariff_excl  = float(np.sum(C_CAP * sim['P_peak_m']))
    elec_net_excl    = opex_buy_excl - opex_sell_excl + cap_tariff_excl + FIXED_ELEC_ANNUAL
    elec_net_incl    = elec_net_excl * (1 + VAT)

    # ------------------------------------------------------------------
    # Totals
    # ------------------------------------------------------------------
    total_annual_cost = capex_total_ann + maint_total + gas_cost_incl + elec_net_incl

    return {
        # CAPEX breakdown
        'capex_boiler_ann' : capex_boiler_ann,
        'capex_pv_ann'     : capex_pv_ann,
        'capex_tes_ann'    : capex_tes_ann,
        'capex_total_ann'  : capex_total_ann,
        # Maintenance
        'maint_boiler'     : maint_boiler * (1 + VAT),
        'maint_pv'         : maint_pv     * (1 + VAT),
        'maint_tes'        : maint_tes    * (1 + VAT),
        'maint_total'      : maint_total,
        # Gas
        'gas_annual_kwh'   : gas_annual_kwh,
        'co2_annual_tonne' : co2_annual_tonne,
        'ets2_cost'        : ets2_cost_incl,
        'gas_commodity_cost': (gas_annual_kwh * GAS_PRICE_EUR_KWH) * (1 + VAT),
        'gas_cost'         : gas_cost_incl,
        # Electricity
        'elec_buy'         : opex_buy_excl  * (1 + VAT),
        'elec_sell'        : opex_sell_excl * (1 + VAT),
        'cap_tariff'       : cap_tariff_excl * (1 + VAT),
        'fixed_elec'       : FIXED_ELEC_ANNUAL * (1 + VAT),
        'elec_net'         : elec_net_incl,
        # Totals
        'total_annual_cost': total_annual_cost,
    }


# ==============================================================================
# --- 7. PRINT REPORT
# ==============================================================================

def print_report(sim, costs):
    rem_boiler = remaining_life(LIFETIME_GAS_BOILER, AGE_INSTALLATION)
    rem_pv     = remaining_life(LIFETIME_PV,         AGE_INSTALLATION)
    rem_tes    = remaining_life(LIFETIME_TES,         AGE_INSTALLATION)

    print("\n" + "=" * 65)
    print("  REFERENCE COST — EXISTING INSTALLATION (Outsider Viewpoint)")
    print("=" * 65)
    print(f"\n  Installation age : {AGE_INSTALLATION} years")
    print(f"  Discount rate    : {DISCOUNT_RATE*100:.1f} %")
    print(f"  VAT              : {VAT*100:.0f} %")

    print(f"\n--- ASSETS ---")
    print(f"  Gas boilers  : {N_BOILERS} × {BOILER_UNIT_KW:.0f} kW_th = {BOILER_TOTAL_KW:.0f} kW_th"
          f"  (η={ETA_BOILER*100:.0f}%,  remaining life {rem_boiler} yr)")
    print(f"  PV           : {PV_INSTALLED_KWP:.0f} kWp"
          f"                    (remaining life {rem_pv} yr)")
    print(f"  TES (water)  : {TES_INSTALLED_LITRES:.0f} L"
          f"                    (remaining life {rem_tes} yr)")

    print(f"\n--- CAPEX (annualised, outsider viewpoint, excl. VAT) ---")
    print(f"  Gas boilers  : €{costs['capex_boiler_ann']:>10,.0f}/yr   "
          f"  (CRF over {rem_boiler} yr, CAPEX = €{CAPEX_BOILER_UNIT*BOILER_TOTAL_KW:,.0f})")
    print(f"  PV           : €{costs['capex_pv_ann']:>10,.0f}/yr   "
          f"  (CRF over {rem_pv} yr, CAPEX = €{CAPEX_PV_UNIT*PV_INSTALLED_KWP:,.0f})")
    print(f"  TES (water)  : €{costs['capex_tes_ann']:>10,.0f}/yr   "
          f"  (CRF over {rem_tes} yr, CAPEX = €{CAPEX_TES_EUR_LITRE*TES_INSTALLED_LITRES:,.0f})")
    print(f"  ─────────────────────────")
    print(f"  Total CAPEX  : €{costs['capex_total_ann']:>10,.0f}/yr")

    print(f"\n--- OPEX — Maintenance (incl. {VAT*100:.0f}% VAT) ---")
    print(f"  Gas boilers  : €{costs['maint_boiler']:>10,.0f}/yr")
    print(f"  PV           : €{costs['maint_pv']:>10,.0f}/yr")
    print(f"  TES          : €{costs['maint_tes']:>10,.0f}/yr")
    print(f"  ─────────────────────────")
    print(f"  Total maint. : €{costs['maint_total']:>10,.0f}/yr")

    print(f"\n--- OPEX — Gas (incl. {VAT*100:.0f}% VAT) ---")
    print(f"  Annual gas consumed      : {costs['gas_annual_kwh']:>10,.0f} kWh_gas/yr")
    print(f"  Annual CO2 emitted       : {costs['co2_annual_tonne']:>10,.1f} tonne CO2/yr")
    print(f"  Gas commodity price      : {GAS_PRICE_EUR_KWH*100:.2f} ct/kWh (Fluvius Imewo, all-in excl. VAT)")
    print(f"  Gas variable cost        :               →  €{costs['gas_commodity_cost']:>10,.0f}/yr (incl. VAT)")
    print(f"  Gas fixed costs          : {FIXED_GAS_ANNUAL:.2f} €/yr  →  €{FIXED_GAS_ANNUAL*(1+VAT):>10,.0f}/yr (incl. VAT)")
    if ETS2_ENABLED:
        print(f"  ETS2 carbon cost         : {ETS2_PRICE_EUR_TONNE:.0f} €/tonne CO2"
              f"  →  €{costs['ets2_cost']:>10,.0f}/yr (incl. VAT)")
        print(f"    [ETS2: {CO2_FACTOR_KG_KWH_GAS:.3f} kg CO2/kWh × {costs['gas_annual_kwh']:,.0f} kWh"
              f" = {costs['co2_annual_tonne']:.1f} t CO2 × {ETS2_PRICE_EUR_TONNE:.0f} €/t]")
    else:
        print(f"  ETS2 carbon cost         : DISABLED")
    print(f"  ─────────────────────────")
    print(f"  Gas cost total (incl. VAT): €{costs['gas_cost']:>10,.0f}/yr")

    print(f"\n--- OPEX — Electricity (incl. {VAT*100:.0f}% VAT) ---")
    print(f"  Annual grid buy          : {sim['P_buy'].sum():>10,.0f} kWh   "
          f" →  €{costs['elec_buy']:>10,.0f}/yr")
    print(f"  Annual PV injection      : {sim['P_sell'].sum():>10,.0f} kWh   "
          f" →  -€{costs['elec_sell']:>10,.0f}/yr")
    print(f"  Annual PV generation     : {sim['PV_gen'].sum():>10,.0f} kWh")
    print(f"  Capacity tariff          :                €{costs['cap_tariff']:>10,.0f}/yr")
    print(f"    Monthly peaks [kW] : {[f'{p:.1f}' for p in sim['P_peak_m']]}")
    print(f"  Fixed connection cost    :                €{costs['fixed_elec']:>10,.0f}/yr")
    print(f"  ─────────────────────────")
    print(f"  Net electricity cost     : €{costs['elec_net']:>10,.0f}/yr")

    print(f"\n{'=' * 65}")
    print(f"  TOTAL ANNUAL COST (reference / old system) :")
    print(f"    CAPEX (annualised)   : €{costs['capex_total_ann']:>10,.0f}/yr")
    print(f"    Maintenance          : €{costs['maint_total']:>10,.0f}/yr")
    ets2_label = f" (incl. ETS2 €{costs['ets2_cost']:,.0f}/yr)" if ETS2_ENABLED else " (ETS2 disabled)"
    print(f"    Gas OPEX             : €{costs['gas_cost']:>10,.0f}/yr{ets2_label}")
    print(f"    Electricity OPEX     : €{costs['elec_net']:>10,.0f}/yr")
    print(f"  ─────────────────────────")
    print(f"    TOTAL                : €{costs['total_annual_cost']:>10,.0f}/yr")
    print("=" * 65)

    # --- Comparison with MILP result ---
    try:
        with open('milp_opt.json') as f:
            milp = json.load(f)
        milp_total = milp['obj']
        saving = costs['total_annual_cost'] - milp_total
        saving_pct = saving / costs['total_annual_cost'] * 100
        print(f"\n  COMPARISON WITH MILP OPTIMISED SYSTEM:")
        print(f"    MILP total cost          : €{milp_total:>10,.0f}/yr")
        print(f"    Reference (old system)   : €{costs['total_annual_cost']:>10,.0f}/yr")
        if saving >= 0:
            print(f"    Annual saving (new vs old): €{saving:>9,.0f}/yr  ({saving_pct:.1f}% reduction)")
        else:
            print(f"    Annual cost increase (new vs old): €{-saving:>9,.0f}/yr  ({-saving_pct:.1f}% more expensive)")
        print("=" * 65)
    except FileNotFoundError:
        print("\n  (milp_opt.json not found — run the MILP optimisation first to compare.)")


# ==============================================================================
# --- 8. MAIN
# ==============================================================================

if __name__ == "__main__":
    print(f"=== Baseline Reference Cost — Existing Installation ({YEAR}) ===\n")

    # Load shared data (same sources as the MILP model)
    fetch_pvgis_to_csv(50, 60)   # temp_c=50°C, temp_h=60°C (return / supply)
    I_SOLAR, _COP, _T_amb = load_pvgis_from_csv()

    if PRICE_SOURCE == "epex":
        P_price_buy, P_price_sell = load_prices_from_epex(EPEX_CSV_PATH)
    else:
        P_price_buy, P_price_sell = load_prices_from_csv()

    P_THERMAL_LOAD = load_demand_from_gascsv(GAS_CSV_PATH, GAS_DEMAND_YEAR)
    P_LOAD         = load_electrical_demand_from_meter(METER_CSV_PATH, ELEC_DEMAND_YEAR)

    # Simulate the existing installation
    sim   = simulate_old_installation(I_SOLAR, P_THERMAL_LOAD, P_LOAD,
                                      P_price_buy, P_price_sell)

    # Calculate costs
    costs = calculate_reference_costs(sim, P_price_buy, P_price_sell)

    # Print report
    print_report(sim, costs)
