import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(n, original_df):
    """
    Generates n synthetic samples using realistic ranges and physical relationships,
    including random indoor temperature that influences building load and energy consumption.
    """
    # Time generation
    if "Local Time (Timezone : GMT+8h)" in original_df.columns:
        last_time = pd.to_datetime(original_df["Local Time (Timezone : GMT+8h)"].iloc[-1])
    else:
        last_time = datetime.now()
    times = [last_time + timedelta(hours=i+1) for i in range(n)]
    
    # Temperature ranges
    outside_temp = np.random.uniform(-20, 100, n)  # °F
    indoor_temp = np.random.uniform(64, 79, n)     # °F
    
    # Cooling water temperature influenced by indoor temp
    cooling_water_temp = np.random.uniform(0, 10, n) - (indoor_temp - 70) * 0.05
    
    # Chilled Water Rate (L/sec)
    if "Chilled Water Rate (L/sec)" in original_df.columns:
        min_rate = original_df["Chilled Water Rate (L/sec)"].min()
        max_rate = original_df["Chilled Water Rate (L/sec)"].max()
    else:
        min_rate, max_rate = 1.0, 10.0
    chilled_water_rate = np.random.uniform(min_rate, max_rate, n)
    
    # Humidity calculation based on dew point and outside temp
    dew_point = np.random.uniform(30, 70, n)  # °F
    es_dew = 6.112 * np.exp((17.62 * dew_point) / (243.12 + dew_point))
    es_temp = 6.112 * np.exp((17.62 * outside_temp) / (243.12 + outside_temp))
    hum = 100 * es_dew / es_temp
    
    # Wind Speed and Pressure
    wind_speed = np.random.uniform(0, 20, n)
    pressure = np.random.uniform(28, 31, n)
    
    # Building load influenced by indoor and outside temperature
    building_load = 300 + 0.8 * (indoor_temp - 70) + 2.5 * (outside_temp - indoor_temp) + np.random.normal(0, 10, n)
    
    # Chiller energy consumption influenced by building load and indoor-outdoor temp difference
    chiller_energy = 500 + 0.2 * building_load + 0.05 * (outside_temp - indoor_temp) + np.random.normal(0, 20, n)
    
    # Combine into DataFrame
    synthetic_df = pd.DataFrame({
        "Local Time (Timezone : GMT+8h)": times,
        "Desired Indoor Temperature (F)": indoor_temp,
        "Outside Temperature (F)": outside_temp,
        "Cooling Water Temperature (C)": cooling_water_temp,
        "Chilled Water Rate (L/sec)": chilled_water_rate,
        "Building Load (RT)": building_load,
        "Chiller Energy Consumption (kWh)": chiller_energy,
        "Dew Point (F)": dew_point,
        "Humidity (%)": hum,
        "Wind Speed (mph)": wind_speed,
        "Pressure (in)": pressure
    })
    
    return synthetic_df

# --- Main Program ---

# Specify the path to your original data file (update the path accordingly)
input_file = "HVAC Energy Data.csv"  # Change to your actual file path

# Load the original data
original_df = pd.read_csv(input_file)

# Display a preview of the original data
print("Original Data Preview:")
print(original_df.head())

# Generate synthetic data (e.g., 1000 new samples) with more diverse temperature values
n_new_samples = 20000
synthetic_df = generate_synthetic_data(n_new_samples, original_df)

# Append the synthetic data to the original dataset
augmented_df = synthetic_df

# Save the augmented dataset to a new CSV file
output_file = "augmented_dataset.csv"
augmented_df.to_csv(output_file, index=False)
print(f"Augmented dataset with {n_new_samples} synthetic samples saved to '{output_file}'.")
