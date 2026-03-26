# ============================================================
# fetch_be_prices.py
# Fetches Belgian ENTSO-E day-ahead electricity prices
# for a given date and returns a 24-value €/kWh array.
#
# Requirements:  pip install entsoe-py pandas matplotlib
#
# API key setup:
#   1. Register at https://transparency.entsoe.eu
#   2. Email transparency@entsoe.eu  — subject: "Restful API access"
#      body: your registered email address
#   3. After approval, go to Account Settings → generate your token
#   4. Paste it in ENTSOE_API_KEY below (or use an env variable)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from entsoe import EntsoePandasClient

# ── Configuration ────────────────────────────────────────────
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY",'1477b98a-948f-4ea8-8194-f87af65dd4f4')
COUNTRY_CODE   = "BE"          # Belgium bidding zone
TIMEZONE       = "Europe/Brussels"

def fetch_yearly_day_ahead_prices_be(year: int) -> pd.DataFrame:
    """
    Fetch Belgian ENTSO-E day-ahead electricity prices for an entire year.

    Parameters
    ----------
    year : int
        The year to fetch, e.g. 2023

    Returns
    -------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex (hourly, Europe/Brussels timezone)
        and columns:
            'price_eur_per_mwh'  — price in €/MWh
            'price_eur_per_kwh'  — price in €/kWh
        Missing hours are forward-filled then back-filled.
        Falls back to NaN rows for any chunk that fails after retrying.
    """
    import time

    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)

    start = pd.Timestamp(f"{year}-01-01", tz=TIMEZONE)
    end   = pd.Timestamp(f"{year+1}-01-01", tz=TIMEZONE)

    # ENTSO-E recommends fetching in monthly chunks to avoid timeouts
    monthly_chunks = pd.date_range(start, end, freq="MS", tz=TIMEZONE)
    all_series = []

    for i, chunk_start in enumerate(monthly_chunks[:-1]):
        chunk_end = monthly_chunks[i + 1]
        month_label = chunk_start.strftime("%Y-%m")

        for attempt in range(1, 4):          # up to 3 retries
            try:
                prices = client.query_day_ahead_prices(
                    COUNTRY_CODE, start=chunk_start, end=chunk_end
                )
                prices_kwh = prices / 1000.0

                # Normalise to hourly resolution
                hourly = (
                    prices_kwh
                    .resample("1h")
                    .mean()
                )
                all_series.append(hourly)
                print(f"  ✅ {month_label}  ({len(hourly)} hours)")
                time.sleep(0.5)              # be polite to the API
                break

            except Exception as e:
                print(f"  ⚠  {month_label} attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    time.sleep(5 * attempt)
                else:
                    print(f"     → Inserting NaN block for {month_label}")
                    # placeholder so the index stays continuous
                    idx = pd.date_range(chunk_start, chunk_end,
                                        freq="1h", tz=TIMEZONE, inclusive="left")
                    all_series.append(pd.Series(np.nan, index=idx))

    # Concatenate and build a clean full-year hourly index
    full_index = pd.date_range(start, end, freq="1h",
                               tz=TIMEZONE, inclusive="left")
    combined = pd.concat(all_series).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.reindex(full_index)

    # Fill small gaps (DST transitions, missing quarter-hours, etc.)
    combined = combined.ffill(limit=3).bfill(limit=3)

    df = pd.DataFrame({
        "price_eur_per_kwh": combined,
        "price_eur_per_mwh": combined * 1000,
    })

    total      = len(df)
    nan_count  = df["price_eur_per_kwh"].isna().sum()
    valid      = total - nan_count
    print(f"\n📊 Year {year}: {valid}/{total} valid hours "
          f"({nan_count} NaN), "
          f"mean {df['price_eur_per_mwh'].mean():.2f} €/MWh")
    return df


def fetch_day_ahead_prices_be(date_str: str) -> np.ndarray:
    """
    Fetch Belgian ENTSO-E day-ahead electricity prices for one day.

    Parameters
    ----------
    date_str : str
        Date in 'YYYY-MM-DD' format, e.g. '2023-07-15'

    Returns
    -------
    prices_eur_per_kwh : np.ndarray, shape (24,)
        Hourly prices in €/kWh (index 0 = 00:00 local time).
        Falls back to a synthetic price profile on failure.
    """
    target = pd.Timestamp(date_str, tz=TIMEZONE)
    start  = target
    end    = target + pd.Timedelta(days=1)

    try:
        client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
        prices = client.query_day_ahead_prices(
            COUNTRY_CODE, start=start, end=end
        )

        # The API returns prices in €/MWh → convert to €/kWh
        prices_kwh = prices / 1000.0

        # Resample to exactly 24 hourly values (handles 15-min resolution)
        prices_hourly = (
            prices_kwh
            .resample("1h")
            .mean()
            .reindex(pd.date_range(start, periods=24, freq="1h", tz=TIMEZONE))
            .interpolate(method="time")   # fill any gaps
        )

        result = prices_hourly.values
        print(f"✅ Fetched {len(result)} hourly prices for {date_str} (Belgium)")
        print(f"   Min: {result.min():.4f}  Max: {result.max():.4f}  "
              f"Mean: {result.mean():.4f}  €/kWh")
        return result

    except Exception as e:
        print(f"⚠  ENTSO-E fetch failed: {e}")
        print("   → Using synthetic fallback prices.")
        return _synthetic_fallback()


def _synthetic_fallback() -> np.ndarray:
    """Typical Belgian day-ahead price shape (€/kWh) as fallback."""
    return np.array([
        0.08, 0.07, 0.07, 0.07, 0.08, 0.10,
        0.14, 0.18, 0.16, 0.12, 0.10, 0.09,
        0.09, 0.09, 0.10, 0.12, 0.16, 0.22,
        0.24, 0.20, 0.15, 0.11, 0.09, 0.08,
    ])


def plot_prices(prices: np.ndarray, date_str: str) -> None:
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(hours, prices * 1000, where="post", color="steelblue", lw=2)
    ax.fill_between(hours, prices * 1000, step="post", alpha=0.2, color="steelblue")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Price [€/MWh]")
    ax.set_title(f"Belgian Day-Ahead Electricity Prices — {date_str}")
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    df_2023 = fetch_yearly_day_ahead_prices_be(2023)
    print(df_2023.head(48))

    # Optional: save to CSV
    df_2023.to_csv("be_prices_2023.csv")

    # Optional: plot the full year
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(df_2023.index, df_2023["price_eur_per_mwh"],
            lw=0.4, color="steelblue")
    ax.set_ylabel("€/MWh")
    ax.set_title("Belgian Day-Ahead Prices 2023")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()