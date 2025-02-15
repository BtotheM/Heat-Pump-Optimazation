import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


#READ IMPORTANT
#needed to complete app: 
#-training/testing set for ML model
#-visualisation of model performance(accuracy of the actual model(absolute error, squared error, absolute error))
#-Show case the model by running multiple simulations where the temperature ranges, then using visualisation methods(scatter plot, graph ect) in order to show the 
#efficacy of the application
#-calculation helper function needs to be finished: READ COMMENT INSIDE FUNCTION

st.title("Random Forest Regressor model: Predict the best Chilled Water Temperature point depending on varying factors")
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
    data["heating_load"] = ((24 - data["external_temp"]) * 2.2 + (data["Humidity (%)"] / 12) + (data["occupancy"] * 5) + np.sin(data["external_temp"] / 15) * 3 + noise)

    return data

#Application
uploaded_file = st.file_uploader("Upload CSV dataset (required).", type=["csv"])
if uploaded_file is None:
    st.warning("Upload a dataset to procced")
    st.stop()

data = pd.read_csv(uploaded_file)
data = CreateSyntheticCols(data)

st.dataframe(data.head(130000))

@st.cache_resource
#Random Forest Regression model that takes the rows in the featured variable and used them to predict the best CWT. 
#This returns a trained model that can predict the best CWT from future data. 
#Important variables for predicting CWT depending on corr: Temp, Building Load, Chiller Energy consumption, heating_load, dew point, occupancy
#Important variables for predicting heating_load depending on corr: chilled_water_temp, occupancy, external_temp, Chiller Energy Consumption (kWh), Building Load (RT), Dew Point (F)

def train_heating_load_model(df):
    features = ["external_temp", "Humidity (%)" ,"occupancy", "Building Load (RT)", "Chiller Energy Consumption (kWh)", "heating_load"]
    X = df[features]
    y = df["chilled_water_temp"]
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=16,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X, y)
    return model

model = train_heating_load_model(data)

# Plant Tonnage input
st.header("Plant Specifications")
Toninput = st.number_input("Enter Plant Tonnage in Tons:", value=200, min_value=50, max_value=1000)

# Temperature ranges for predicted data.
EXTtemps = np.linspace(-5, 35, 41)     
baseline_COP_value = 3.75
default_humidity = 50.0
default_occupancy = 3
noise = np.random.normal(7, 5)

temp = st.number_input("What is the current temperature(°C)", value= 10.0, step=1.0)
occ = st.number_input("How many occupants are in the building?", value= 2.0, step= 1.0)
hum = st.number_input("What is the current humidity?",value= 40.0, step = 5.0)
heatingload = (((24 - temp * 2.2 + hum / 12) + occ * 5) + np.sin(temp / 15) * 3 + noise)
buildingload = 300 + 0.8*(occ) + 2.5*(temp - 24) + np.random.normal(0, 10)
chillerengergy = 500 + 0.2*(buildingload) + np.random.normal(0, 20)
#"external_temp", "Humidity (%)" ,"occupancy", "chilled_water_temp, Building Load (RT), Chiller Energy Consumption (kWh), heating_load" needed for prediction

PredVars = pd.DataFrame({
    "external_temp": [temp],
    "Humidity (%)": [hum],
    "occupancy": [occ],
    "Building Load (RT)": [buildingload],
    "Chiller Energy Consumption (kWh)": [chillerengergy],
    "heating_load": [heatingload]
})
#made a helper function so simulations with varied tempetures can be run later 
def helpercalculations(temp,pred, HeatingLoad): 
    optimizedpowerinput, eta = heat_pump_efficiency(temp, pred, HeatingLoad)
    baselinepowerinput = heatingload / baseline_COP_value
    EFFgain = ((baselinepowerinput - optimizedpowerinput) / baselinepowerinput) * 100
    st.write(f"Our baseline power per ton is {baselinepowerinput}, which is calculated by dividing the heating load by a constant baseline COP value. In our case the baseline COP value is 3.75. After our model chooses the optimal CWT point our optimized power input is {optimizedpowerinput} and our efficiency gain being {EFFgain}" )
    #needed calculations: baseline total power, optimized total power, total savings
    return 

prediction = model.predict(PredVars)[0]
st.write(f"The best predicted CWT: {prediction}" )

helpercalculations(temp,prediction, heatingload)



