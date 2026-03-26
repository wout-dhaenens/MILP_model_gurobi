import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator

# --- Configuration ---
file_path = r'C:\Users\woutd\OneDrive - UGent\Bureaublad\2e master\thesis\Plot20.12.xlsx'
plt.style.use('seaborn-v0_8-whitegrid')

# ── DAY SELECTOR ──────────────────────────────────────────────────────────────
# Set the date you want to plot (YYYY-MM-DD format)
PLOT_DATE = '2025-12-16'   # <-- change this to any date in the dataset
# ─────────────────────────────────────────────────────────────────────────────

# Constants for Energy Calculation
FLOW_RATE_M3H = 8.0
DENSITY_WATER = 1000
CP_WATER = 4.186

try:
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'VarValue_low': 'Temperature Low (3u4)',
        'VarValue_high': 'Temperature High (3u2)'
    })
    df['TimeString'] = pd.to_datetime(df['TimeString'], format='%d.%m.%Y %H:%M:%S')
    df = df.set_index('TimeString')

    # --- Power Calculation (on full dataset before filtering) ---
    df['Delta_T'] = df['Temperature High (3u2)'] - df['Temperature Low (3u4)']
    mass_flow_rate = (FLOW_RATE_M3H * DENSITY_WATER) / 3600
    df['Power_kW'] = mass_flow_rate * CP_WATER * df['Delta_T'] * df['VarValue_pump']
    
    # --- Filter to selected day ---
    target_date = pd.Timestamp(PLOT_DATE)
    day_start = target_date
    day_end   = target_date + pd.Timedelta(days=1)
    df_day = df.loc[day_start:day_end]

    if df_day.empty:
        available = df.index.normalize().unique().strftime('%Y-%m-%d').tolist()
        raise ValueError(
            f"No data found for '{PLOT_DATE}'.\n"
            f"Available dates in dataset: {available}"
        )

    date_label = target_date.strftime('%d %B %Y')
    print(f"Plotting data for: {date_label}  ({len(df_day)} rows)")

    # ── PLOT 1: Full 3-subplot overview ──────────────────────────────────────
    fig1, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, figsize=(14, 12), sharex=True,
        gridspec_kw={'height_ratios': [2, 1, 1.5]}
    )

    ax1.plot(df_day.index, df_day['Temperature Low (3u4)'],  color='#1f77b4', lw=1.5, label='Temp Low')
    ax1.plot(df_day.index, df_day['Temperature High (3u2)'], color='#d62728', lw=1.5, label='Temp High')
    ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax1.set_title(f'System Performance & Energy Consumption — {date_label}',
                  fontsize=15, pad=15, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True)

    ax2.step(df_day.index, df_day['VarValue_pump'], color='#2ca02c', lw=2, where='post')
    ax2.fill_between(df_day.index, 0, df_day['VarValue_pump'], step='post', alpha=0.2, color='#2ca02c')
    ax2.set_ylabel('Pump', fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['OFF', 'ON'])

    ax3.plot(df_day.index, df_day['Power_kW'], color='#ff7f0e', lw=1.5, label='Power (kW)')
    ax3.fill_between(df_day.index, 0, df_day['Power_kW'], alpha=0.3, color='#ff7f0e')
    ax3.set_ylabel('Power (kW)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time', fontsize=12, fontweight='bold')

    for ax in (ax1, ax2, ax3):
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

    plt.xticks(rotation=45)
    plt.tight_layout()
    out1 = f'overview_{PLOT_DATE}.png'
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out1}")

    # ── PLOT 2: Standalone Power plot ─────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_day.index, df_day['Power_kW'], color='#ff7f0e', lw=1.5, label='Power (kW)')
    ax.fill_between(df_day.index, 0, df_day['Power_kW'], alpha=0.3, color='#ff7f0e')
    ax.set_ylabel('Power (kW)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_title(f'Instantaneous Thermal Power — {date_label}',
                 fontsize=14, pad=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.xaxis.set_major_locator(AutoDateLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    out2 = f'power_{PLOT_DATE}.png'
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out2}")

    # --- Daily energy total ---
    time_diff_hours = df_day.index.to_series().diff().dt.total_seconds() / 3600
    daily_energy_kwh = (df_day['Power_kW'] * time_diff_hours).sum()
    print(f"\nTotal thermal energy on {date_label}: {daily_energy_kwh:.2f} kWh")

except Exception as e:
    print(f"Error: {e}")