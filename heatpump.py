#Important variables for predicting CWT depending on corr: Temp, Building Load, Chiller Energy consumption, heating_load, dew point, occupancy
#Important variables for predicting heating_load depending on corr: chilled_water_temp, occupancy, external_temp, Chiller Energy Consumption (kWh), Building Load (RT), Dew Point (F)  import streamlit as st
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.title("Random Forest Regression Model: Optimizing Chilled Water temperature points depending on multiple factors")
def heat_pump_efficiency(external_temp, chilled_water_temp, heating_demand):
    """
    Calculates the power input using a Carnot COP model with an adaptive efficiency factor.
    T_hot is fixed at 313.15 K (40°C) and T_cold is derived from the chilled water temperature.
    """
    T_hot = 313.15  # 40°C in Kelvin
    T_cold = max(chilled_water_temp + 273.15, 273.16)
    temperature_difference = T_hot - T_cold
    eta = np.clip(0.85 - 0.003 * max(temperature_difference - 5, 0), 0.55, 0.85)
    COP_carnot = T_hot / (T_hot - T_cold) if (T_hot - T_cold) > 0 else 1
    COP_real = max(COP_carnot * eta, 1.5)
    power_input = heating_demand / COP_real
    return power_input, eta

def CreateSyntheticCols(data):
    n = len(data)
    offset = np.random.normal(loc=0, scale=1, size=n)
    noise = np.random.normal(7, 5, n)
    data["external_temp"] = (data["Outside Temperature (F)"] - 32) * 5/9
    data["occupancy"] = np.random.randint(0, 10, n)
    building_load = data["Building Load (RT)"] 
    data["chilled_water_temp"] = (15 - 0.1 * (data["external_temp"] - 20) - 0.005 * building_load - 0.2 * data["occupancy"] + offset)
    data["heating_load"] = ((24 - data["external_temp"]) * 2.2 + (data["Humidity (%)"] / 12) + (data["occupancy"] * 5) + np.sin(data["external_temp"] / 15) * 3 + 0.1 * data["chilled_water_temp"] + noise)
    data["heating_load"] = data["heating_load"].abs()
    return data

#Application
uploaded_file = st.file_uploader("Upload CSV dataset.", type=["csv"])
if uploaded_file is None:
    st.stop()

data = pd.read_csv(uploaded_file)
data = CreateSyntheticCols(data)

st.dataframe(data.head(100000))

#Random Forest Regression model that takes the temperature humidity occupancy and CWT then as features and uses the given heating load data as a target. 
#This returns a trained model that can predict the heating loads from future data temp, humidity, occupancy and CWT. 

def train_heating_load_model(df):

    features = ["external_temp", "Humidity (%)" ,"occupancy", "chilled_water_temp","Building Load (RT)", "Chiller Energy Consumption (kWh)"]
    X = df[features]
    y = df["heating_load"]
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size= .2, random_state=19)
    model = RandomForestRegressor(
        n_estimators= 100,
        max_depth= 20,
        min_samples_split= 5,
        min_samples_leaf= 2,
        random_state=13
    )
    model.fit(Xtrain, Ytrain)
    y_pred = model.predict(Xtest)
    st.write(mean_absolute_error(y_pred, Ytest), mean_squared_error(y_pred, Ytest), r2_score(y_pred, Ytest))

    return model

model = train_heating_load_model(data)

# Plant Tonnage input
st.header("Plant Specifications")
Toninput = st.number_input("Enter Plant Tonnage in Tons:", value=1000, min_value=50, max_value=50000)
temp_option = st.radio("Choose an option:", ["Temperature Range Simulation", "Specific Temperature Simulation"])
# search for best CWT using grid search
st.header("2. Grid Search Over Chilled Water Temperature")


baseline_COP_value = 3.75
#["external_temp", "Humidity (%)" ,"occupancy", "chilled_water_temp, Dew Point (F), Chiller Energy Consumption (kWh), Building Load (RT)"]
def calculations(temp, dewpoint):
    EXTtemp = temp
    dewpoint = dewpoint
    es_dew = 6.112 * np.exp((17.62 * dewpoint) / (243.12 + dewpoint))
    es_temp = 6.112 * np.exp((17.62 * EXTtemp) / (243.12 + EXTtemp))
    hum = 100 * es_dew / es_temp
    occ = np.random.randint(0, 10)
    buildingload = 300 + 0.8*(occ) + 2.5*(EXTtemp - 24) + np.random.normal(0, 10)
    chillerengergy = 500 + 0.2*(buildingload) + np.random.normal(0, 20)
    return EXTtemp, dewpoint, hum, occ, buildingload, chillerengergy
Electricity_Cost_per_kWh = 0.12
all_results = []
CWTtemps = np.linspace(8,14, 31)
if temp_option == "Temperature Range Simulation":
    all_results = []
    for _ in range(50):
        EXTtemp = np.random.uniform(5, 35)
        dewpoint = np.random.uniform(5, 30)
        
        # Use the calculations() function for generating humidity, occupancy, and loads
        EXTtemp, dewpoint, hum, occ, buildingload, chillerenergy = calculations(EXTtemp, dewpoint)
        
        # Loop over chilled water temperature values
        for CWTtemp in CWTtemps:
            PredVars = pd.DataFrame({
                "external_temp": [EXTtemp],
                "Humidity (%)": [hum],
                "occupancy": [occ],
                "chilled_water_temp": [CWTtemp],
                "Building Load (RT)": [buildingload],
                "Chiller Energy Consumption (kWh)": [chillerenergy]
            })

            # Keep only the required columns
            PredVars = PredVars[["external_temp", "Humidity (%)", "occupancy", "chilled_water_temp", 
                                 "Building Load (RT)", "Chiller Energy Consumption (kWh)"]]
            
            # Predict heating load
            predicted_load = model.predict(PredVars)[0]
            optimized_power, eta = heat_pump_efficiency(EXTtemp, CWTtemp, predicted_load)
            baseline_power = predicted_load / baseline_COP_value
            efficiency_gain = ((baseline_power - optimized_power) / baseline_power) * 100
            
            # Store results
            all_results.append({
                "external_temp": EXTtemp,
                "chilled_water_temp": CWTtemp,
                "predicted_load_per_ton": predicted_load,
                "optimized_power_per_ton": optimized_power,
                "baseline_power_per_ton": baseline_power,
                "efficiency_gain": efficiency_gain
            })

    # Convert results to DataFrame
    all_results = pd.DataFrame(all_results)
    st.dataframe(all_results)
    # Group and find the best chilled water temperature for each external temperature
    OptimizedCWT = all_results.groupby("external_temp", as_index=False).apply(lambda g: g.loc[g["optimized_power_per_ton"].idxmin()]).reset_index(drop=True)

    # Display results in Streamlit
    st.write("### Best Chilled Water Temp for Each Outside Temperature")
    OptimizedCWT["total_baseline_power"] = OptimizedCWT["baseline_power_per_ton"] * Toninput
    OptimizedCWT["total_optimized_power"] = OptimizedCWT["optimized_power_per_ton"] * Toninput
    OptimizedCWT["total_savings"] = OptimizedCWT["total_baseline_power"] - OptimizedCWT["total_optimized_power"]
    st.dataframe(OptimizedCWT)

    # Plot Best CWT vs. Outside Temperature
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(OptimizedCWT["external_temp"], OptimizedCWT["chilled_water_temp"], marker="o", color="blue")
    ax1.set_xlabel("Outside Temperature (°C)")
    ax1.set_ylabel("Best Chilled Water Temperature (°C)")
    ax1.set_title("Best CWT vs. Outside Temperature")
    st.pyplot(fig1)

    # Plot Optimized vs. Baseline Power
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(OptimizedCWT["external_temp"], OptimizedCWT["optimized_power_per_ton"], color="blue", label="Optimized Power")
    ax2.scatter(OptimizedCWT["external_temp"], OptimizedCWT["baseline_power_per_ton"], color="red", label="Baseline Power")
    ax2.set_xlabel("Outside Temperature (°C)")
    ax2.set_ylabel("Power (kW per ton)")
    ax2.legend()
    st.pyplot(fig2)

if temp_option == "Specific Temperature Simulation":
  
    all_results = []
    CWTtemps = np.linspace(4,30, 31)
    temp = st.number_input("Enter temperature in Celsius: ", value=10, min_value=-50, max_value=100)
    dewpoint = st.number_input("Enter Dew Point: ", value=25, min_value=-50, max_value=100)
    hum = st.number_input("Enter Humidity: ", value=50, min_value=-50, max_value=100)
    occ = st.number_input("Enter Occupants in the building: ", value=1, min_value=1, max_value=15)
    maxocc = occ
    EXTtemp, dewpoint, hum, occ1, buildingload, chillerenergy = calculations(temp, dewpoint)

    for x in range(24):
    
        for CWTtemp in CWTtemps:
            PredVars = pd.DataFrame({
                "external_temp": [EXTtemp],
                "Humidity (%)": [hum],
                "occupancy": [occ],
                "chilled_water_temp": [CWTtemp],
                "Building Load (RT)": [buildingload],
                "Chiller Energy Consumption (kWh)": [chillerenergy]
            })

            # Keep only the required columns
            PredVars = PredVars[["external_temp", "Humidity (%)", "occupancy", "chilled_water_temp", 
                                            "Building Load (RT)", "Chiller Energy Consumption (kWh)"]]
            
            # Predict heating load
            predicted_load = model.predict(PredVars)
            optimized_power, eta = heat_pump_efficiency(EXTtemp, CWTtemp, predicted_load)
            baseline_power = predicted_load / baseline_COP_value
            efficiency_gain = ((baseline_power - optimized_power) / baseline_power) * 100
            total_savings = (baseline_power - optimized_power)
            total_savings_dollars = total_savings * Electricity_Cost_per_kWh  

        # Store results
            all_results.append({
                "Hour of the day": x,
                "external_temp": EXTtemp,
                "Occupants": occ,
                "chilled_water_temp": CWTtemp,
                "predicted_load_per_ton (kWh)": predicted_load,
                "optimized_power_per_ton (kWh)": optimized_power,
                "baseline_power_per_ton (kWh)": baseline_power,
                "efficiency_gain": efficiency_gain,
                "Total_savings (kWh)": total_savings,
                "Total_savings_dollars": total_savings_dollars,
            })

        
        occ = np.random.randint(1, 15)
        

    all_results = pd.DataFrame(all_results)
    all_results = all_results.groupby("Hour of the day", as_index=False).apply(lambda g: g.loc[g["optimized_power_per_ton (kWh)"].idxmin()]).reset_index(drop=True)
    st.dataframe(all_results)
    HrVSDollars, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(all_results["Hour of the day"], all_results["Total_savings_dollars"], marker="o", color="red", label = "Hours VS Total Savings")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Dollar Savings")
    ax1.set_title("Daily savings with differing occupants per hour")
    st.pyplot(HrVSDollars)

    

    


