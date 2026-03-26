import numpy as np
import pandas as pd
import requests
import math

# ==============================================================================
# --- PARAMETERS (match these to your main optimization script) ---
# ==============================================================================

YEAR       = 2023
T          = 8760
LAT        = 50.85
LON        = 4.35
TILT       = 35
AZIMUTH    = -90
PEAK_POWER = 1.0   # kWp (normalized — main script scales by C_PV)

temp_h, temp_c = 70, 50

# Output file paths
PVGIS_CSV  = "pvgis_data.csv"
PRICES_CSV = "prices_data.csv"


# ==============================================================================
# --- HELPERS ---
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


# ==============================================================================
# --- FETCH PVGIS (solar irradiance + ambient temperature → COP) ---
# ==============================================================================

def fetch_pvgis_to_csv(lat, lon, year, tilt, azimuth, peak_power, out_path):
    print(f"Fetching PVGIS data for lat={lat}, lon={lon}, year={year}...")
    url = (f"https://re.jrc.ec.europa.eu/api/seriescalc?"
           f"lat={lat}&lon={lon}"
           f"&startyear={year}&endyear={year}"
           f"&pvcalculation=1&peakpower={peak_power}"
           f"&angle={tilt}&aspect={azimuth}"
           f"&loss=14&outputformat=json")

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data['outputs']['hourly'])
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
    df = df[df['time'].dt.year == year].reset_index(drop=True)

    if len(df) != T:
        raise ValueError(f"Expected {T} rows from PVGIS, got {len(df)}.")

    df['pv_kw_per_kwp'] = df['P'] / 1000.0
    df['T2m_C']         = df['T2m']
    df['cop']           = df['T2m'].apply(
        lambda t: calculate_lorenz_cop(temp_c, temp_h, t))

    out = df[['time', 'pv_kw_per_kwp', 'T2m_C', 'cop']].copy()
    out.to_csv(out_path, index=False)
    print(f"  Saved {len(out)} rows to '{out_path}'.")
    print(f"  PV range:  {out['pv_kw_per_kwp'].min():.3f} – {out['pv_kw_per_kwp'].max():.3f} kW/kWp")
    print(f"  Temp range: {out['T2m_C'].min():.1f} – {out['T2m_C'].max():.1f} °C")
    print(f"  COP range:  {out['cop'].min():.2f} – {out['cop'].max():.2f}")
    return out


# ==============================================================================
# --- FETCH DAY-AHEAD PRICES (ENTSO-E via public transparency platform) ---
# ==============================================================================

def fetch_entsoe_prices_to_csv(year, out_path):
    """
    Tries to load prices from a local 'load_data.csv' first (same logic as
    main script), then falls back to a realistic synthetic BE price profile.
    Either way the result is saved to out_path so the main script never needs
    to call the API again.
    """
    prices = None


    # --- Attempt 2: fetch_be_prices helper module ---
    if prices is None:
        try:
            from fetch_be_prices import fetch_yearly_day_ahead_prices_be  # type: ignore
            result = fetch_yearly_day_ahead_prices_be(year)
            prices_candidate = np.array(result, dtype=float).flatten()
            if len(prices_candidate) >= T:
                prices = prices_candidate[:T]
                print("  Prices loaded via fetch_be_prices module.")
        except Exception as e:
            print(f"  fetch_be_prices module unavailable: {e}")

    # --- Fallback: realistic synthetic profile ---
    if prices is None:
        print("  Using synthetic day-ahead price profile.")
        fallback_day = np.array(
            [0.10]*4 + [0.12, 0.15, 0.40, 0.50, 0.40, 0.20, 0.15] +
            [0.10]*5 + [0.15, 0.30, 0.50, 0.60, 0.40, 0.20] + [0.10]*3
        )
        prices = np.zeros(T)
        for h in range(T):
            day       = h // 24
            hour      = h % 24
            season    = 1.0 + 0.3 * np.cos(2 * np.pi * day / 365)
            prices[h] = fallback_day[hour] * season

    # Save
    prices = np.array(prices, dtype=float).flatten()[:T]  # ensure 1-D
    dates  = pd.date_range(f"{year}-01-01", periods=T, freq='h')
    out    = pd.DataFrame({'time': dates, 'price_eur_kwh': prices})
    out.to_csv(out_path, index=False)
    print(f"  Saved {len(out)} rows to '{out_path}'.")
    print(f"  Price range: {prices.min():.4f} – {prices.max():.4f} €/kWh")
    print(f"  Annual avg:  {prices.mean():.4f} €/kWh")
    return out


# ==============================================================================
# --- LOADERS (used by the main optimization script instead of API calls) ---
# ==============================================================================

def load_pvgis_from_csv(path=PVGIS_CSV):
    """Returns (pv_kw_per_kwp, cop) as numpy arrays of length 8760."""
    df = pd.read_csv(path, parse_dates=['time'])
    assert len(df) == T, f"Expected {T} rows in {path}, got {len(df)}"
    return df['pv_kw_per_kwp'].to_numpy(dtype=float), df['cop'].to_numpy(dtype=float)


def load_prices_from_csv(path=PRICES_CSV):
    """Returns price array as numpy array of length 8760."""
    df = pd.read_csv(path, parse_dates=['time'])
    assert len(df) == T, f"Expected {T} rows in {path}, got {len(df)}"
    return df['price_eur_kwh'].to_numpy(dtype=float)


# ==============================================================================
# --- MAIN ---
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(f"  Data fetcher — year {YEAR}, location ({LAT}, {LON})")
    print("=" * 60)

    print(f"\n[1/2] PVGIS solar + COP data → '{PVGIS_CSV}'")
    fetch_pvgis_to_csv(LAT, LON, YEAR, TILT, AZIMUTH, PEAK_POWER, PVGIS_CSV)

    print(f"\n[2/2] Day-ahead electricity prices → '{PRICES_CSV}'")
    fetch_entsoe_prices_to_csv(YEAR, PRICES_CSV)

    print("\nDone! You can now load both files in your optimization script:")
    print("  from fetch_and_save_data import load_pvgis_from_csv, load_prices_from_csv")
    print("  I_SOLAR, COP_t = load_pvgis_from_csv()")
    print("  P_price        = load_prices_from_csv()")
