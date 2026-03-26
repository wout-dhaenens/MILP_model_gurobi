"""
generate_heat_profile.py
------------------------
Standalone BDEW heat demand profile generator for Belgium, commercial buildings.
Replicates the when2heat pipeline using local temperature data.

Inputs:
    - hourly temperature CSV (same format as temperatures_Merelbeke_2024_2025.xlsx)
    - daily_demand.csv     (BDEW sigmoid parameters)
    - hourly_factors_COM.csv (BGW hourly profile factors)

Output:
    - year_heat_profile_COM.csv  (8760 rows: datetime + normalized heat demand)

Usage:
    Edit the file paths at the bottom, then run: python generate_heat_profile.py
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────
# 1. LOAD TEMPERATURE DATA
# ─────────────────────────────────────────────
def load_temperature(filepath):
    """
    Load hourly temperature data. Expects two columns: datetime and temperature (°C).
    Adjust column indices below if your file has a different structure.
    """
    ext = filepath.split('.')[-1].lower()
    if ext in ('xlsx', 'xls'):
        df = pd.read_excel(filepath)
    else:
        # Try common separators
        df = pd.read_csv(filepath, sep=None, engine='python', decimal=',')

    # Use first two columns: datetime and temperature
    df = df.iloc[:, :2].copy()
    df.columns = ['datetime', 'T_C']
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    df['T_C'] = pd.to_numeric(df['T_C'], errors='coerce')
    df = df.dropna()
    return df


# ─────────────────────────────────────────────
# 2. DAILY MEAN → REFERENCE TEMPERATURE
# ─────────────────────────────────────────────
def reference_temperature(hourly_temp_C):
    """
    Weighted mean of today + 3 preceding days (BDEW thermal inertia correction).
    Weights: today=0.5, yesterday=0.25, 2 days ago=0.125, 3 days ago=0.125
    Returns daily series in Kelvin (as used in when2heat).
    """
    daily_K = hourly_temp_C.resample('D').mean() + 273.15
    weights = [0.5, 0.25, 0.125, 0.125]
    ref = (
        weights[0] * daily_K
        + weights[1] * daily_K.shift(1)
        + weights[2] * daily_K.shift(2)
        + weights[3] * daily_K.shift(3)
    )
    # Fill leading NaNs with the raw daily mean
    ref = ref.fillna(daily_K)
    return ref  # Kelvin


# ─────────────────────────────────────────────
# 3. HEATING THRESHOLD ADJUSTMENT (Belgium)
# ─────────────────────────────────────────────
# Belgium = 15.2°C, Germany (BDEW baseline) = 13.98°C → offset = +1.22 K
# Source: heating_thresholds.csv (Kozarcanin et al. 2019, via when2heat)
BELGIUM_THRESHOLD_OFFSET_K = 15.2 - 13.98  # = 1.22

def adjust_temperature(ref_temp_K):
    """Shift reference temperature by Belgium–Germany heating threshold difference."""
    return ref_temp_K + BELGIUM_THRESHOLD_OFFSET_K


# ─────────────────────────────────────────────
# 4. BDEW SIGMOID → DAILY DEMAND FACTOR
# ─────────────────────────────────────────────
def load_daily_parameters(filepath):
    """Parse daily_demand.csv into a dict of COM (normal windiness) parameters."""
    df = pd.read_csv(filepath, sep=';', index_col=0, decimal=',', header=[0, 1])
    # Columns are MultiIndex: (building_type, windiness)
    # We want COM, normal
    params = df[('COM', 'normal')].to_dict()
    return params


def daily_heat_factor(adjusted_temp_K, params):
    """
    BDEW sigmoid function for daily space heating demand.
    h(T) = A / (1 + (B / (T_C - T0))^C) + D
    where T0 is implicit in the sigmoid shape (effectively 40°C offset).
    
    Note: when2heat uses temperature in Kelvin internally but the sigmoid
    parameters are calibrated in Celsius, so we convert back.
    """
    T_C = adjusted_temp_K - 273.15  # back to Celsius for sigmoid

    A = params['A']
    B = params['B']
    C = params['C']
    D = params['D']

    # Sigmoid: avoid division by zero when T_C == 40
    denominator = 1 + (B / (T_C - 40.0)) ** C
    h = A / denominator + D

    # Clip to non-negative
    h = h.clip(lower=0)
    return h  # dimensionless daily demand factor


# ─────────────────────────────────────────────
# 5. BGW HOURLY FACTORS → HOURLY PROFILE
# ─────────────────────────────────────────────
def load_hourly_factors(filepath):
    """
    Parse hourly_factors_COM.csv.
    Returns DataFrame indexed by (weekday 0-6, hour 0-23),
    columns = temperature bins [-15,-10,-5,0,5,10,15,20,25,30].
    """
    df = pd.read_csv(filepath, sep=';', decimal=',')
    df.columns = ['weekday', 'time'] + [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
    df['hour'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour
    df = df.drop(columns='time')
    df = df.set_index(['weekday', 'hour'])
    return df


def interpolate_hourly_factor(hourly_factors, weekday, hour, T_C):
    """
    Look up and interpolate the hourly factor for a given weekday, hour, and temperature.
    Uses linear interpolation between the two nearest temperature bins.
    """
    temp_bins = np.array([-15, -10, -5, 0, 5, 10, 15, 20, 25, 30])
    T_clipped = np.clip(T_C, temp_bins[0], temp_bins[-1])
    row = hourly_factors.loc[(weekday, hour)].values.astype(float)
    return np.interp(T_clipped, temp_bins, row)


def hourly_heat_profile(daily_factor, ref_temp_K, hourly_factors):
    """
    Distribute daily heat demand across hours using BGW lookup table.
    Returns normalized hourly heat demand series (dimensionless).
    """
    # Upsample daily factor to hourly (forward-fill within each day)
    # Reindex to cover all hours of the year before ffill
    all_hours = pd.date_range(daily_factor.index[0], periods=len(daily_factor)*24, freq='h')
    daily_factor_hourly = daily_factor.resample('h').ffill().reindex(all_hours, method='ffill')

    result = []
    dates = []

    for ts, row in daily_factor_hourly.items():
        weekday = ts.weekday()  # 0=Monday ... 6=Sunday
        hour = ts.hour
        date = ts.date()

        # Daily reference temperature for this day (in Celsius)
        date_str = str(date)
        idx_str = ref_temp_K.index.astype(str)
        if date_str in idx_str:
            val = ref_temp_K.iloc[list(idx_str).index(date_str)]
            T_ref_C = (val.item() if hasattr(val, 'item') else float(val)) - 273.15
        else:
            T_ref_C = 10.0

        # Hourly BGW factor (interpolated by temperature)
        f_hour = interpolate_hourly_factor(hourly_factors, weekday, hour, T_ref_C)

        # Hourly demand = daily factor × hourly shape factor × 24
        # (×24 so the hourly profile integrates to daily_factor over the day)
        result.append(row * f_hour * 24)
        dates.append(ts)

    profile = pd.Series(result, index=dates)

    # Normalize so annual sum = 1.0 (matches when2heat normalization)
    annual_sum = profile.sum()
    if annual_sum > 0:
        profile = profile / annual_sum

    return profile


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────
def generate_profile(temp_filepath, daily_params_filepath, hourly_factors_filepath, output_filepath):
    print("Loading temperature data...")
    temp_df = load_temperature(temp_filepath)
    print(f"  → {len(temp_df)} hourly records from {temp_df.index[0]} to {temp_df.index[-1]}")

    print("Computing reference temperature...")
    ref_temp_K = reference_temperature(temp_df['T_C'])

    print(f"Applying Belgium heating threshold offset ({BELGIUM_THRESHOLD_OFFSET_K:+.1f} K)...")
    adj_temp_K = adjust_temperature(ref_temp_K)

    print("Loading BDEW parameters (COM, normal windiness)...")
    params = load_daily_parameters(daily_params_filepath)
    print(f"  → A={params['A']:.4f}, B={params['B']:.4f}, C={params['C']:.4f}, D={params['D']:.4f}")

    print("Computing daily heat demand factors (BDEW sigmoid)...")
    daily_factor = daily_heat_factor(adj_temp_K, params)

    print("Loading BGW hourly profile factors...")
    hourly_factors = load_hourly_factors(hourly_factors_filepath)

    print("Distributing demand across hours (BGW lookup)...")
    profile = hourly_heat_profile(daily_factor, adj_temp_K, hourly_factors)

    # Trim to exactly 8760 hours if needed
    if len(profile) > 8760:
        profile = profile.iloc[:8760]

    print(f"Saving output to {output_filepath}...")
    out_df = pd.DataFrame({
        'datetime': profile.index,
        'heat_demand_normalized': profile.values
    })
    out_df.to_csv(output_filepath, index=False)
    print(f"Done. {len(out_df)} rows written.")
    print(f"  Annual sum (should be ~1.0): {profile.sum():.4f}")
    print(f"  Peak hour demand: {profile.max():.6f}  (multiply by annual kWh to get kW)")

    return profile


# ─────────────────────────────────────────────
# 7. RUN
# ─────────────────────────────────────────────
if __name__ == '__main__':

    # ── Edit these paths ──────────────────────
    TEMP_FILE           = '02_HeatDemand/weather_data/temperatures_Merelbeke_2026.xlsx'
    DAILY_PARAMS_FILE   = '02_HeatDemand/daily_demand.csv'
    HOURLY_FACTORS_FILE = '02_HeatDemand/hourly_factors_COM.csv'
    OUTPUT_FILE         = '02_HeatDemand/year_heat_profile_COM_2026.csv'
    # ─────────────────────────────────────────

    profile = generate_profile(
        temp_filepath           = TEMP_FILE,
        daily_params_filepath   = DAILY_PARAMS_FILE,
        hourly_factors_filepath = HOURLY_FACTORS_FILE,
        output_filepath         = OUTPUT_FILE,
    )

    # Optional: scale to a known annual demand in kWh
    # annual_demand_kWh = 500_000  # e.g. 500 MWh/year
    # Q_kW = profile * annual_demand_kWh  # hourly kW series