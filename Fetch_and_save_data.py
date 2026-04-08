import numpy as np
import pandas as pd
import requests
import math
import matplotlib.pyplot as plt
from fetch_be_prices import fetch_yearly_day_ahead_prices_be


# ==============================================================================
# --- PARAMETERS ---
# ==============================================================================

YEAR        = 2023
T           = 8760
LAT         = 51.18
LON         = 3.55
TILT        = 35
AZIMUTH     = 10
PEAK_POWER  = 1.0        # kWp, normalized
BUY_MARKUP  = 0.083       # EUR/kWh added on top of spot for buy price
SELL_MARKUP = 0.03       # EUR/kWh added on top of spot for sell price


PVGIS_CSV   = "pvgis_data.csv"
PRICES_CSV  = "prices_data.csv"


# ==============================================================================
# --- HELPERS ---
# ==============================================================================


# Calibrated from EWYE050CZNAA2 manufacturer data (CustomPoints export):
# 6 stable load points at T_amb=20°C, T_supply=55°C, Q=28–63 kW
# η_Lorenz = COP_actual / COP_Lorenz_ideal → mean=0.361, std=0.011
ETA_LORENZ = 0.361

def calculate_lorenz_cop(t_sink_in, t_sink_out, t_source_in):
    T_in  = t_sink_in  + 273.15
    T_out = t_sink_out + 273.15
    T_src = t_source_in + 273.15
    if T_in == T_out:
        T_h_avg = T_in
    else:
        T_h_avg = (T_out - T_in) / math.log(T_out / T_in)
    cop_lorenz_ideal = T_h_avg / (T_h_avg - T_src)
    return ETA_LORENZ * cop_lorenz_ideal


# ==============================================================================
# --- FETCH & SAVE PVGIS ---
# ==============================================================================

def fetch_pvgis_to_csv(temp_c=50,temp_h=70):
    print(f"Fetching PVGIS data for lat={LAT}, lon={LON}, year={YEAR}...")
    url = (f"https://re.jrc.ec.europa.eu/api/seriescalc?"
           f"lat={LAT}&lon={LON}"
           f"&startyear={YEAR}&endyear={YEAR}"
           f"&pvcalculation=1&peakpower={PEAK_POWER}"
           f"&angle={TILT}&aspect={AZIMUTH}"
           f"&loss=14&outputformat=json")

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data['outputs']['hourly'])
    # PVGIS timestamps are UTC — convert to Brussels local time
    df['time'] = (pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
                    .dt.tz_localize('UTC')
                    .dt.tz_convert('Europe/Brussels')
                    .dt.tz_localize(None)           # strip tz info for clean CSV
                    .dt.floor('h'))                 # remove the :10 min offset
    # Keep only rows that fall within the target year after tz conversion,
    # then reindex to a clean hourly Brussels-time grid (handles DST edge rows)
    start_local = pd.Timestamp(f"{YEAR}-01-01")
    end_local   = pd.Timestamp(f"{YEAR+1}-01-01")
    df = df[(df['time'] >= start_local) & (df['time'] < end_local)].reset_index(drop=True)
    # Drop duplicate timestamps caused by DST fall-back (keep first occurrence)
    df = df.drop_duplicates(subset='time', keep='first').set_index('time')
    full_index = pd.date_range(start_local, end_local, freq='h', inclusive='left')
    df = df.reindex(full_index)
    df.index.name = 'time'
    df[['P', 'T2m']] = df[['P', 'T2m']].ffill().bfill()
    df = df.reset_index()

    if len(df) != T:
        raise ValueError(f"Expected {T} rows from PVGIS, got {len(df)}.")

    df['pv_kw_per_kwp'] = df['P'] / 1000.0
    df['T2m_C']         = df['T2m']
    df['cop']           = df['T2m'].apply(
        lambda t: calculate_lorenz_cop(temp_c, temp_h, t))

    out = df[['time', 'pv_kw_per_kwp', 'T2m_C', 'cop']]
    out.to_csv(PVGIS_CSV, index=False)
    print(f"  Saved {len(out)} rows to '{PVGIS_CSV}'.")
    print(f"  PV  : {out['pv_kw_per_kwp'].min():.3f} - {out['pv_kw_per_kwp'].max():.3f} kW/kWp")
    print(f"  Temp: {out['T2m_C'].min():.1f} - {out['T2m_C'].max():.1f} deg C")
    print(f"  COP : {out['cop'].min():.2f} - {out['cop'].max():.2f}")


# ==============================================================================
# --- FETCH & SAVE PRICES ---
# ==============================================================================

def fetch_prices_to_csv():
    print(f"\nFetching BE day-ahead prices for {YEAR}...")

    # fetch_yearly_day_ahead_prices_be returns a DataFrame with
    # 'price_eur_per_kwh' column and a tz-aware DatetimeIndex
    df_raw = fetch_yearly_day_ahead_prices_be(YEAR)

    spot = df_raw['price_eur_per_kwh'].to_numpy(dtype=float).flatten()[:T]

    if len(spot) != T:
        raise ValueError(f"Expected {T} price values, got {len(spot)}.")

    dates = pd.date_range(f"{YEAR}-01-01", periods=T, freq='h')
    out   = pd.DataFrame({
        'time':         dates,
        'spot_eur_kwh': spot,
        'buy_eur_kwh':  1.02 * spot + BUY_MARKUP,          # spot + grid fees/taxes
        'sell_eur_kwh': 0.98 * spot - SELL_MARKUP,       
    })
    out.to_csv(PRICES_CSV, index=False)
    print(f"  Saved {len(out)} rows to '{PRICES_CSV}'.")
    print(f"  Spot : {spot.min():.4f} - {spot.max():.4f} EUR/kWh  (avg {spot.mean():.4f})")
    print(f"  Buy  : {(spot+BUY_MARKUP).min():.4f} - {(spot+BUY_MARKUP).max():.4f} EUR/kWh  (spot + {BUY_MARKUP} markup)")
    print(f"  Sell : {np.maximum(spot,0).min():.4f} - {np.maximum(spot,0).max():.4f} EUR/kWh  (spot only)")


# ==============================================================================
# --- LOADERS (use these in your optimization script) ---
# ==============================================================================

# ==============================================================================
# --- LOADERS (use these in your optimization script) ---
# ==============================================================================

def load_pvgis_from_csv(path=PVGIS_CSV):
    """Returns (pv_kw_per_kwp, cop, t_amb) as numpy arrays of length 8760."""
    df = pd.read_csv(path, parse_dates=['time'])
    assert len(df) == T, f"Expected {T} rows in {path}, got {len(df)}"
    
    # Extract the three arrays
    pv_kw = df['pv_kw_per_kwp'].to_numpy(dtype=float)
    cop = df['cop'].to_numpy(dtype=float)
    t_amb = df['T2m_C'].to_numpy(dtype=float)
    
    return pv_kw, cop, t_amb

def load_prices_from_csv(path=PRICES_CSV):
    """Returns (buy_price, sell_price) as numpy arrays of length 8760 in EUR/kWh."""
    df = pd.read_csv(path, parse_dates=['time'])
    assert len(df) == T, f"Expected {T} rows in {path}, got {len(df)}"
    return (df['buy_eur_kwh'].to_numpy(dtype=float),
            df['sell_eur_kwh'].to_numpy(dtype=float))


EPEX_CSV = "epex_2025.csv"

def load_prices_from_epex(path=EPEX_CSV,
                          buy_markup=BUY_MARKUP,
                          sell_markup=SELL_MARKUP):
    """
    Loads day-ahead spot prices from an EPEX CSV (Belgian format).

    Expected columns: Date (DD/MM/YYYY), Time (e.g. '14u'), Euro (e.g. '€ 84.65')
    Prices in the file are in EUR/MWh — they are converted to EUR/kWh here.
    Buy price  = spot + buy_markup   (markup in EUR/kWh, e.g. grid fees, taxes)
    Sell price = spot + sell_markup  (injection tariff offset)

    Returns (buy_price, sell_price) as numpy arrays of length T in EUR/kWh,
    sorted chronologically (ascending).
    """
    df = pd.read_csv(path)

    # --- Parse price: strip '€', non-breaking spaces, convert €/MWh -> €/kWh ---
    df['spot_eur_kwh'] = (
        df['Euro']
        .str.replace('€', '', regex=False)
        .str.replace('\xa0', '', regex=False)   # non-breaking space
        .str.strip()
        .astype(float)
        / 1000.0
    )

    # --- Parse datetime from Date (DD/MM/YYYY) + Time ('14u') ---
    df['hour'] = df['Time'].str.replace('u', '', regex=False).astype(int)
    df['datetime'] = pd.to_datetime(df['Date'], dayfirst=True) + pd.to_timedelta(df['hour'], unit='h')

    # --- Sort ascending and reset index ---
    df = df.sort_values('datetime').reset_index(drop=True)

    assert len(df) == T, (
        f"EPEX file has {len(df)} rows, expected {T}. "
        f"Check that the file covers exactly one full year."
    )

    spot = df['spot_eur_kwh'].to_numpy(dtype=float)
    buy  = spot + buy_markup          # same as prices_data.csv
    sell = 0.98 * spot - sell_markup  # same as prices_data.csv: 2% DSO injection fee + markup

    print(f"  EPEX prices loaded from '{path}'")
    print(f"  Period : {df['datetime'].iloc[0]}  →  {df['datetime'].iloc[-1]}")
    print(f"  Spot   : mean={spot.mean()*1000:.1f} €/MWh  "
          f"min={spot.min()*1000:.1f}  max={spot.max()*1000:.1f}")

    return buy, sell
# ==============================================================================
# --- PLOTTING ---
# ==============================================================================

def plot_prices(path=PRICES_CSV, start_hour=0, end_hour=168):
    """
    Plots the spot, buy, and sell prices from the saved CSV.
    Defaults to the first 168 hours (1 week) for readability.
    Set end_hour=T (8760) to plot the entire year.
    """
    try:
        df = pd.read_csv(path, parse_dates=['time'])
    except FileNotFoundError:
        print(f"File '{path}' not found. Please run fetch_prices_to_csv() first.")
        return

    # Slice the dataframe
    df_slice = df.iloc[start_hour:end_hour]

    plt.figure(figsize=(12, 6))
    
    # Plotting the three price curves
    plt.plot(df_slice['time'], df_slice['buy_eur_kwh'], label='Buy Price (EUR/kWh)', color='crimson', linewidth=1.5)
    plt.plot(df_slice['time'], df_slice['sell_eur_kwh'], label='Sell Price (EUR/kWh)', color='forestgreen', linewidth=1.5)
    plt.plot(df_slice['time'], df_slice['spot_eur_kwh'], label='Spot Price (EUR/kWh)', color='royalblue', linestyle='--', alpha=0.7)

    plt.title(f"Electricity Prices in Belgium (Hours {start_hour} to {end_hour})")
    plt.xlabel("Date / Time")
    plt.ylabel("Price (EUR/kWh)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plt.show()

# ==============================================================================
# --- MAIN ---
# ==============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print(f"  Fetching data for {YEAR}  |  ({LAT}, {LON})")
    print(f"  Buy markup: {BUY_MARKUP} EUR/kWh")
    print("=" * 55)

    fetch_pvgis_to_csv()
    fetch_prices_to_csv()
    plot_prices()
    print("\nDone! In your optimization script use:")
    print("  from fetch_and_save_data import load_pvgis_from_csv, load_prices_from_csv")
    print("  I_SOLAR, COP_t            = load_pvgis_from_csv()")
    print("  P_buy_price, P_sell_price = load_prices_from_csv()")
    
    
    
    
    