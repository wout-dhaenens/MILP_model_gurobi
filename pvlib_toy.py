import pandas as pd
import matplotlib.pyplot as plt
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# 1. Setup Location (Eeklo, Belgium)
lat, lon = 51.1835, 3.5476
location = Location(latitude=lat, longitude=lon, tz='UTC')

# 2. Fetch TMY Data (Scientific standard for sizing - Lund et al., 2017)
# Unpacking 2 values as per modern pvlib versions
# Use keyword arguments to ensure the email and variables are handled correctly
weather, meta = pvlib.iotools.get_pvgis_tmy(lat, lon, map_variables=True)
# 3. Define the PV System (Fixed 'pdc' key for PVWatts)
system = PVSystem(
    surface_tilt=35, 
    surface_azimuth=180,
    module_parameters={'pdc0': 5000, 'gamma_pdc': -0.004}, 
    inverter_parameters={'pdc0': 5000}, # Corrected key
    temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
)

# 4. ModelChain setup
mc = ModelChain(system, location, spectral_model='no_loss', 
                ac_model='pvwatts', aoi_model='no_loss')

# 5. Run for the full year to allow for seasonal analysis
mc.run_model(weather)

# 6. Extract results
results = pd.DataFrame(index=weather.index)
results['pv_kw'] = mc.results.ac / 1000.0
results['temp_amb'] = weather['temp_air']

# 7. Plotting Function for January
def plot_january_data(df):
    # Slice for January (first 31 days)
    jan_data = df.iloc[0:744]
    
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Axis 1: PV Power
    ax1.set_xlabel('Hour of Year (January)')
    ax1.set_ylabel('PV Power Output [kW]', color='tab:blue')
    ax1.fill_between(jan_data.index, jan_data['pv_kw'], color='tab:blue', alpha=0.3, label='PV Power')
    ax1.plot(jan_data.index, jan_data['pv_kw'], color='tab:blue', lw=1.5)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 5.5)

    # Axis 2: Ambient Temperature
    ax2 = ax1.twinx()
    ax2.set_ylabel('Ambient Temperature [°C]', color='tab:red')
    ax2.plot(jan_data.index, jan_data['temp_amb'], color='tab:red', lw=1.5, label='Ambient Temp')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Horizontal line at 0°C for reference
    ax2.axhline(0, color='black', lw=0.8, ls='--')

    plt.title('January Design Data for Eeklo (TMY Basis)')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

# Run the plotting function
plot_january_data(results)