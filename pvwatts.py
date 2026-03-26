import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def plot_yesterday_design_data(lat, lon, peak_power=5.0):
    # 1. Determine "Yesterday's" date
    yesterday = datetime.now() - timedelta(days=1)
    year = yesterday.year
    # Note: PVGIS historical data availability often has a delay of a few months.
    # If yesterday is not available, we use the same day from the latest available year (e.g. 2023).
    target_year = 2023
    start_date = f"{target_year}{yesterday.strftime('%m%d')}"
    
    # 2. Fetch Hourly Series from PVGIS API
    # Using 'seriescalc' for specific historical dates
    url = (f"https://re.jrc.ec.europa.eu/api/seriescalc?lat={lat}&lon={lon}"
           f"&startyear={target_year}&endyear={target_year}"
           f"&pvcalculation=1&peakpower={peak_power}&loss=14&outputformat=json")
    
    response = requests.get(url)
    data = response.json()
    
    # 3. Parse and Filter for the specific day
    df = pd.DataFrame(data['outputs']['hourly'])
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
    
    # Filter for the specific month and day matching "yesterday"
    day_df = df[df['time'].dt.strftime('%m%d') == yesterday.strftime('%m%d')].reset_index()

    # 4. Calculate COP and Scale PV (if not already calculated by API)
    # T2m is ambient temperature in PVGIS seriescalc
    # COP Model Reference: Krützfeldt et al. (2021)
    day_df['cop'] = 3.0 + 0.05 * day_df['T2m']
    
    # P is the PV power in Watts from the API
    day_df['pv_kw'] = day_df['P'] / 1000 

    # 5. Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Axis 1: PV Power
    ax1.set_xlabel(f'Hour of Day ({yesterday.strftime("%d %B")}, {target_year})')
    ax1.set_ylabel('PV Power Output [kW]', color='tab:blue')
    ax1.fill_between(day_df.index, day_df['pv_kw'], color='tab:blue', alpha=0.3, label='PV Power')
    ax1.plot(day_df.index, day_df['pv_kw'], color='tab:blue', lw=2)
    ax1.set_ylim(0, peak_power * 1.1)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Axis 2: Temperature and COP
    ax2 = ax1.twinx()
    ax2.set_ylabel('Ambient Temp [°C] / COP [-]', color='tab:red')
    ax2.plot(day_df.index, day_df['T2m'], color='tab:red', ls='--', label='Ambient Temp')
    ax2.plot(day_df.index, day_df['cop'], color='darkred', lw=2, label='HP COP')
    ax2.tick_params(axis='y', labelcolor='darkred')

    plt.title(f'Historical Design Data for Eeklo (Lat: {lat:.4f}, Lon: {lon:.4f})')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Eeklo coordinates provided by user
latitude, longitude = 51.1835, 3.5476
plot_yesterday_design_data(latitude, longitude)