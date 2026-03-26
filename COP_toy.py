import math
import pandas as pd

def calculate_lorenz_cop(t_sink_in, t_sink_out, t_source_in):
    """
    Calculates Lorenz COP using the thermodynamic logarithmic average temperature.
    Temperatures in Celsius.
    """
    # Convert to Kelvin
    T_in = t_sink_in + 273.15
    T_out = t_sink_out + 273.15
    T_source = t_source_in + 273.15
    
    # Calculate Logarithmic Mean Temperature for the sink
    if T_in == T_out:
        T_h_avg = T_in
    else:
        T_h_avg = (T_out - T_in) / math.log(T_out / T_in)
    
    # Lorenz COP formula
    return T_h_avg / (T_h_avg - T_source)

# Data extracted from your image
model_data = [
    {"cap_mild": 19.0, "cop_mild": 3.40, "cap_cold": 13.4, "cop_cold": 1.39},
    {"cap_mild": 23.5, "cop_mild": 3.36, "cap_cold": 21.2, "cop_cold": 1.76},
    {"cap_mild": 27.6, "cop_mild": 3.28, "cap_cold": 23.6, "cop_cold": 1.73},
    {"cap_mild": 32.0, "cop_mild": 3.36, "cap_cold": 25.6, "cop_cold": 1.78},
    {"cap_mild": 34.9, "cop_mild": 3.21, "cap_cold": 27.1, "cop_cold": 1.75},
    {"cap_mild": 47.2, "cop_mild": 3.32, "cap_cold": 42.5, "cop_cold": 1.75},
    {"cap_mild": 55.3, "cop_mild": 3.25, "cap_cold": 47.3, "cop_cold": 1.72},
    {"cap_mild": 62.0, "cop_mild": 3.13, "cap_cold": 50.3, "cop_cold": 1.71},
    {"cap_mild": 70.0, "cop_mild": 3.17, "cap_cold": 54.3, "cop_cold": 1.75},
]

# Theoretical limits based on the temperature sets in the headers
LORENZ_MILD = calculate_lorenz_cop(40, 45, 7)    # 40/45 at 7°C
LORENZ_COLD = calculate_lorenz_cop(60, 70, -15)  # 60/70 at -15°C

results = []
for i, m in enumerate(model_data):
    results.append({
        "Model": f"Model {i+1}",
        "Cap (kW) Mild": m['cap_mild'],
        "Actual COP (Mild)": m['cop_mild'],
        "2nd Law Eff (Mild) %": (m['cop_mild'] / LORENZ_MILD) * 100,
        "Cap (kW) Cold": m['cap_cold'],
        "Actual COP (Cold)": m['cop_cold'],
        "2nd Law Eff (Cold) %": (m['cop_cold'] / LORENZ_COLD) * 100
    })

df = pd.DataFrame(results)
print(f"Lorenz COP Limit (Mild): {LORENZ_MILD:.2f}")
print(f"Lorenz COP Limit (Cold): {LORENZ_COLD:.2f}\n")
print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))