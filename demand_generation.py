# -*- coding: utf-8 -*-
"""
BDEW Thermal Demand Profile — Commercial Shop, Belgium
Annual demand : 260 MWh/year
Season window : 9 Dec (SEASON_START_YEAR) → 9 Mar (SEASON_START_YEAR+1), hourly

Weather source: PVGIS seriescalc API (T2m field), matching Fetch_and_save_data.py
Fallback      : open-meteo archive API if PVGIS year not yet available (~2yr lag)

Strategy
--------
demandlib normalises a profile so its SUM over the supplied index = annual_heat_demand.
Because the season straddles two calendar years, we generate TWO full-year
profiles (both scaled to 260 MWh) and then slice + concatenate:
  - year1 profile  →  keep Dec 9 … Dec 31
  - year2 profile  →  keep Jan 1  … Mar 9

To generate a different season, just change SEASON_START_YEAR below.

Dependencies:
    pip install demandlib workalendar requests matplotlib pandas
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
from demandlib import bdew

# ── 1. SETTINGS ──────────────────────────────────────────────────────────────
ANNUAL_DEMAND_MWH = 260          # full-year total, used for BOTH years
SHLP_TYPE         = "GKO"        # Kaufhaus / shop
WIND_CLASS        = 1            # 1 = windy (Belgium)
LATITUDE          = 51.05        # Ghent — adjust to your site
LONGITUDE         = 3.72
TILT              = 35           # panel tilt (only needed for PVGIS PV calc)
AZIMUTH           = 0            # 0 = south-facing

SEASON_START_YEAR = 2025         # Dec 9 of this year → Mar 9 of (this year + 1)


# ── 2. FETCH TEMPERATURE ─────────────────────────────────────────────────────
def fetch_temperature_pvgis(year: int) -> pd.Series:
    """Fetch hourly ambient temperature (T2m) for a full year via PVGIS seriescalc."""
    print(f"  Fetching PVGIS temperature {year} ...")
    url = (
        f"https://re.jrc.ec.europa.eu/api/seriescalc?"
        f"lat={LATITUDE}&lon={LONGITUDE}"
        f"&startyear={year}&endyear={year}"
        f"&pvcalculation=1&peakpower=1"
        f"&angle={TILT}&aspect={AZIMUTH}"
        f"&loss=14&outputformat=json"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data["outputs"]["hourly"])
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M").dt.floor("h")
    df = df[df["time"].dt.year == year].reset_index(drop=True)

    # Keep tz-naive — demandlib does not need timezone awareness
    s = pd.Series(
        data  = df["T2m"].values,
        index = pd.DatetimeIndex(df["time"]),
        name  = "temperature",
        dtype = float,
    )
    s = s[~s.index.duplicated(keep="first")]
    s = s.asfreq("h").ffill()
    return s


def fetch_temperature_openmeteo(year: int) -> pd.Series:
    """Fallback: fetch hourly temperature via open-meteo archive API."""
    print(f"  Fetching open-meteo temperature {year} (fallback) ...")
    today    = pd.Timestamp.now().normalize()
    end_date = min(pd.Timestamp(f"{year}-12-31"), today - pd.Timedelta(days=1))
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&start_date={year}-01-01&end_date={end_date.strftime('%Y-%m-%d')}"
        "&hourly=temperature_2m"
        "&timezone=UTC"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Keep tz-naive
    s = pd.Series(
        data  = data["hourly"]["temperature_2m"],
        index = pd.to_datetime(data["hourly"]["time"]),
        name  = "temperature",
        dtype = float,
    )
    s = s[~s.index.duplicated(keep="first")]
    s = s.asfreq("h").ffill()
    return s


def fetch_temperature(year: int, pad_with_year: int = None) -> pd.Series:
    """Try PVGIS first; fall back to open-meteo if the year is not yet in PVGIS.
    If the resulting series is shorter than a full year (e.g. current year),
    pad the missing months using temperatures from pad_with_year (previous year).
    """
    try:
        s = fetch_temperature_pvgis(year)
    except Exception as e:
        print(f"  PVGIS failed for {year}: {e}")
        print(f"  Falling back to open-meteo ...")
        s = fetch_temperature_openmeteo(year)

    expected_hours = 8784 if _is_leap(year) else 8760
    if len(s) < expected_hours - 24 and pad_with_year is not None:
        # Pad missing months using same calendar months from pad_with_year
        print(f"  Padding {year} ({len(s)} h) to full year using {pad_with_year} temperatures ...")
        try:
            pad_src = fetch_temperature_pvgis(pad_with_year)
        except Exception:
            pad_src = fetch_temperature_openmeteo(pad_with_year)

        last_covered = s.index[-1]
        pad_start    = last_covered + pd.Timedelta(hours=1)
        pad_end      = pd.Timestamp(f"{year}-12-31 23:00")

        # Build filler: same month/day/hour from pad_with_year
        filler_idx  = pd.date_range(pad_start, pad_end, freq="h")
        src_idx     = filler_idx.map(
            lambda t: t.replace(year=pad_with_year)
        )
        filler_vals = pad_src.reindex(src_idx).values
        filler      = pd.Series(filler_vals, index=filler_idx, name="temperature")

        s = pd.concat([s, filler])
        s = s[~s.index.duplicated(keep="first")]
        s = s.asfreq("h").ffill()
        print(f"  Padded series now covers {s.index[0]} to {s.index[-1]} ({len(s)} h)")

    return s


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


# ── 3. BELGIAN PUBLIC HOLIDAYS ────────────────────────────────────────────────
def get_holidays(years: list) -> dict:
    try:
        from workalendar.europe import Belgium
        cal = Belgium()
        holidays = {}
        for year in years:
            for d, name in cal.holidays(year):
                holidays[d] = name
        print(f"  Belgian holidays: {len(holidays)} entries")
        return holidays
    except ImportError:
        print("  workalendar not installed -- no holidays applied.")
        return {}


# ── 4. GENERATE FULL-YEAR BDEW PROFILE ───────────────────────────────────────
def make_profile(temp: pd.Series, year: int, holidays: dict) -> pd.Series:
    profile = bdew.HeatBuilding(
        df_index           = temp.index,
        temperature        = temp,
        annual_heat_demand = ANNUAL_DEMAND_MWH * 1e3,   # kWh
        shlp_type          = SHLP_TYPE,
        building_class     = 0,     # must be 0 for non-residential types
        wind_class         = WIND_CLASS,
        ww_incl            = True,
        holidays           = holidays,
        name               = f"shop_{year}",
    ).get_bdew_profile()
    return profile / 1e3   # MWh per hour


# ── 5. SEASON PROFILE GENERATOR ──────────────────────────────────────────────
def generate_season_profile(start_year: int):
    """
    Generate the heat demand profile for Dec 9 of start_year to Mar 9 of start_year+1.
    Returns (profile, temp_window) as tz-aware pd.Series (Europe/Brussels).
    """
    end_year    = start_year + 1
    slice_start = pd.Timestamp(f"{start_year}-12-09 00:00")
    slice_end   = pd.Timestamp(f"{end_year}-03-09 23:00")

    print(f"\n=== Season {start_year}/{end_year}: {slice_start.date()} -> {slice_end.date()} ===")

    # Temperatures — pad end_year if it's incomplete (e.g. current year)
    print("Fetching weather data ...")
    temp_y1 = fetch_temperature(start_year)
    temp_y2 = fetch_temperature(end_year, pad_with_year=start_year)

    # Holidays
    holidays = get_holidays([start_year, end_year])

    # Full-year BDEW profiles
    print("\nGenerating BDEW profiles ...")
    profile_y1 = make_profile(temp_y1, start_year, holidays)
    profile_y2 = make_profile(temp_y2, end_year,   holidays)
    print(f"  {start_year} full-year total: {profile_y1.sum():.2f} MWh")
    print(f"  {end_year}   full-year total: {profile_y2.sum():.2f} MWh")

    # Slice to season window and concatenate
    cut_y1 = pd.Timestamp(f"{start_year}-12-31 23:00")
    cut_y2 = pd.Timestamp(f"{end_year}-01-01 00:00")

    profile     = pd.concat([profile_y1[slice_start:cut_y1], profile_y2[cut_y2:slice_end]])
    temp_window = pd.concat([temp_y1[slice_start:cut_y1],    temp_y2[cut_y2:slice_end]])

    print(f"\nSliced window : {profile.index[0]} --> {profile.index[-1]}")
    print(f"  Timesteps   : {len(profile)} h  ({len(profile)/24:.1f} days)")
    print(f"  Window total: {profile.sum():.2f} MWh  "
          f"(~{profile.sum()/ANNUAL_DEMAND_MWH*100:.1f}% of annual)")
    print(f"  Peak power  : {profile.max()*1e3:.1f} kW")
    print(f"  Mean power  : {profile.mean()*1e3:.1f} kW")

    return profile, temp_window


# ── 6. SAVE TO CSV ────────────────────────────────────────────────────────────
def save_profile(profile: pd.Series, temp_window: pd.Series, start_year: int) -> str:
    end_year = start_year + 1
    out_csv  = f"shop_heat_demand_{start_year}_{end_year}.csv"
    pd.DataFrame({
        "datetime"        : profile.index,
        "heat_demand_MWh" : profile.values,
        "heat_demand_kWh" : profile.values * 1e3,
        "temperature_C"   : temp_window.reindex(profile.index).values,
    }).to_csv(out_csv, index=False)
    print(f"\nCSV saved --> {out_csv}")
    return out_csv


# ── 7. PLOT ───────────────────────────────────────────────────────────────────
def plot_profile(profile: pd.Series, temp_window: pd.Series, start_year: int):
    end_year = start_year + 1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(profile.index, profile * 1e3, color="#d62728",
             linewidth=0.6, alpha=0.9, label="Heat demand")
    ax1.set_ylabel("Thermal power [kW]")
    ax1.set_title(
        f"BDEW heat demand — Commercial shop, Belgium (SHLP: {SHLP_TYPE})\n"
        f"Annual basis: {ANNUAL_DEMAND_MWH} MWh/yr  |  "
        f"Season {start_year}/{end_year} |  "
        f"Window total: {profile.sum():.1f} MWh"
    )
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2.plot(temp_window.index, temp_window.values, color="#1f77b4",
             linewidth=0.6, alpha=0.9, label="Outdoor temperature (PVGIS)")
    ax2.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Temperature [°C]")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_png = f"shop_heat_demand_{start_year}_{end_year}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved  --> {out_png}")
    plt.show()


# ── 8. MONTHLY SUMMARY ────────────────────────────────────────────────────────
def print_monthly_summary(profile: pd.Series):
    print("\nMonthly totals [MWh] within window:")
    print(profile.resample("ME").sum().rename("MWh").to_string())


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    profile, temp_window = generate_season_profile(SEASON_START_YEAR)
    save_profile(profile, temp_window, SEASON_START_YEAR)
    plot_profile(profile, temp_window, SEASON_START_YEAR)
    print_monthly_summary(profile)
