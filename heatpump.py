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

@st.cache_resource
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
Toninput = st.number_input("Enter Plant Tonnage in Tons:", value=1000, min_value=50, max_value=2000)

# search for best CWT using grid search
st.header("2. 2D Grid Search Over External & Chilled Water Temperature")
CWTtemps = np.linspace(8,14, 31)

baseline_COP_value = 3.75
#["external_temp", "Humidity (%)" ,"occupancy", "chilled_water_temp, Dew Point (F), Chiller Energy Consumption (kWh), Building Load (RT)"]
all_results = []
for x in range(0,40):
    EXTtemp = np.random.uniform(5, 35)
    dewpoint = np.random.uniform(5, 30)
    es_dew = 6.112 * np.exp((17.62 * dewpoint) / (243.12 + dewpoint))
    es_temp = 6.112 * np.exp((17.62 * EXTtemp) / (243.12 + EXTtemp))
    hum = 100 * es_dew / es_temp
    occ = np.random.randint(0, 10)
    buildingload = 300 + 0.8*(occ) + 2.5*(EXTtemp - 24) + np.random.normal(0, 10)
    chillerengergy = 500 + 0.2*(buildingload) + np.random.normal(0, 20)

    for CWTtemp in CWTtemps:
        PredVars = pd.DataFrame({
            "external_temp": [EXTtemp],
            "Humidity (%)": [hum],
            "occupancy": [occ],
            "chilled_water_temp": [CWTtemp],
            "Building Load (RT)": [buildingload],
            "Chiller Energy Consumption (kWh)": [chillerengergy]
        })
        PredVars = PredVars[["external_temp","Humidity (%)", "occupancy", "chilled_water_temp", "Building Load (RT)", "Chiller Energy Consumption (kWh)"]]
        predicted_load = model.predict(PredVars)[0]
        optimized_power, eta = heat_pump_efficiency(EXTtemp, CWTtemp, predicted_load)
        baseline_power = predicted_load / baseline_COP_value
        efficiency_gain = ((baseline_power - optimized_power) / baseline_power) * 100
        all_results.append({
            "external_temp": EXTtemp,
            "chilled_water_temp": CWTtemp,
            "predicted_load_per_ton": predicted_load,
            "optimized_power_per_ton": optimized_power,
            "baseline_power_per_ton": baseline_power,
            "efficiency_gain": efficiency_gain
        })
all_results = pd.DataFrame(all_results)

# Find best CWT for each temperature
grouped = all_results.groupby("external_temp", as_index=False)
# rows with best optimized power input
OptimizedCWT = grouped.apply(lambda g: g.loc[g["optimized_power_per_ton"].idxmin()]).reset_index(drop=True)

st.write("### Best Chilled Water Temp for Each Outside Temperature")

OptimizedCWT["total_baseline_power"] = OptimizedCWT["baseline_power_per_ton"] * Toninput
OptimizedCWT["total_optimized_power"] = OptimizedCWT["optimized_power_per_ton"] * Toninput
OptimizedCWT["total_savings"] = OptimizedCWT["total_baseline_power"] - OptimizedCWT["total_optimized_power"]

st.dataframe(OptimizedCWT)

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(OptimizedCWT["external_temp"], OptimizedCWT["chilled_water_temp"], marker="o", color="blue")
ax1.set_xlabel("Outside Temperature (°C)")
ax1.set_ylabel("Best Chilled Water Temperature (°C)")
ax1.set_title("Best CWT vs. Outside Temperature")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.scatter(OptimizedCWT["external_temp"], OptimizedCWT["optimized_power_per_ton"], color="blue", label="Optimized Power (per ton)")
ax2.scatter(OptimizedCWT["external_temp"], OptimizedCWT["baseline_power_per_ton"], color="red", label="Baseline Power (per ton)")
ax2.set_xlabel("Outside Temperature (°C)")
ax2.set_ylabel("Power (kW per ton)")
ax2.set_title("Outside Temperature vs. Power (Optimized vs. Baseline at Best CWT)")
ax2.legend()
st.pyplot(fig2)
