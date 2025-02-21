import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
st.markdown("""
    <style>
    /* Sidebar gradient */
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #236BB5, #4F95D0);
    }

    /* Scale the entire main content except the sidebar by 25% via a custom wrapper */
    .main-wrapper {
        transform: scale(0.25);
        transform-origin: top left;
    }

    /* Metrics box */
    div[data-testid="metric-container"] {
        background-color: #2A2A3D;
        border-radius: 10px;
        padding: 10px;
        color: #FFFFFF;
        text-align: center;
        margin: 5px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #FF6B6B;
        color: white;
        border-radius: 8px;
        border: none;
        width: 100%;
    }

    /* Tables */
    .stDataFrame {
        background-color: #2A2A3D;
        border-radius: 12px;
    }

    /* Charts */
    .stPlotlyChart {
        background-color: #2A2A3D;
        border-radius: 12px;
    }

    /* Row container for side-by-side charts */
    .chart-row {
        display: flex;
        flex-wrap: wrap;
        gap: 20px; /* space between charts */
        justify-content: center; /* or space-around, space-between, etc. */
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Each chart column in the row */
    .chart-col {
        flex: 1;
        min-width: 300px; /* ensures it doesn't shrink too small */
        max-width: 600px; /* optional limit, remove if you want more responsive behavior */
    }

    </style>
""", unsafe_allow_html=True)

st.title("Random Forest Regression Model: Optimizing Chilled Water Temperature Points")

with st.sidebar:
    st.header("Simulation Settings")
    sim_option = st.radio(
        "Choose a Simulation Type:",
        ["Temperature Range Simulation", "Specific Temperature Simulation"]
    )
    
    Toninput = st.number_input(
        "Enter Plant Tonnage (Tons):",
        value=1000, min_value=50, max_value=50000
    )

    if sim_option == "Specific Temperature Simulation":
        st.write("**Specific Temperature Inputs**")
        temp = st.number_input("Outside Temperature (°C)", value=10, min_value=-50, max_value=100, key="unique_temp")
        indoortemp = st.number_input("Desired Indoor Temperature (°C)", value=22.0, step=1.0, key="temp_specific")
        dewpoint = st.number_input("Dew Point (°C)", value=25, min_value=-50, max_value=100, key="temp_range")
        hum = st.number_input("Humidity (%)", value=50, min_value=0, max_value=100, key="hum")
        occ = st.number_input("Occupants in Building:", value=1, min_value=1, max_value=15, key="occ")
    else:
        st.write("**Range Simulation** will use random data.")

# ------------------------ Heat Pump Efficiency ------------------------
def heat_pump_efficiency(indoortemp, external_temp, chilled_water_temp, heating_demand):
    #Simulation of a heatpump
    T_hot = indoortemp + 273.15
    T_cold = max(chilled_water_temp + 273.15, 273.16)
    temperature_difference = T_hot - T_cold
    eta = np.clip(0.85 - 0.003 * max(temperature_difference - 5, 0), 0.55, 0.85)
    #max possible efficiency a heat pump can achieve when transferring heat from one place to another
    COP_carnot = T_hot / (T_hot - T_cold) if (T_hot - T_cold) > 0 else 1
    #capped COP at 7 as that is the industry standar 
    COP_real = min(max(COP_carnot * eta, 1.5), 7.0)
    power_input = heating_demand / COP_real
    return power_input, eta

# ------------------------ Synthetic Columns ------------------------
def CreateSyntheticCols(data):
    n = len(data)
    offset = np.random.normal(loc=0, scale=1, size=n)
    noise = np.random.normal(7, 5, n)
    data["Desired Indoor Temperature (C)"] = (data["Desired Indoor Temperature (F)"] - 32) * 5/9
    data["external_temp"] = (data["Outside Temperature (F)"] - 32) * 5/9
    data["occupancy"] = np.random.randint(0, 10, n)
    building_load = data["Building Load (RT)"]
    data["chilled_water_temp"] = (
        15
        - 0.1 * (data["external_temp"] - 20)
        - 0.005 * building_load
        - 0.2 * data["occupancy"]
        + offset
    )
    data["heating_load"] = (
        (data["Desired Indoor Temperature (F)"] - data["external_temp"]) * 2.2
        + (data["Humidity (%)"] / 12)
        + (data["occupancy"] * 5)
        + np.sin(data["external_temp"] / 15) * 3
        + 0.1 * data["chilled_water_temp"]
        + noise
    )
    data["heating_load"] = data["heating_load"].abs()
    return data

# ------------------------ CSV Upload ------------------------
uploaded_file = st.file_uploader("Upload CSV dataset.", type=["csv"])
if uploaded_file is None:
    st.stop()

data = pd.read_csv(uploaded_file)
data = CreateSyntheticCols(data)
st.dataframe(data.head(100000))

# Regression model that predicts the heating load y, depending on the X features

def train_heating_load_model(df):
    features = [
        "Desired Indoor Temperature (C)",
        "external_temp",
        "Humidity (%)",
        "occupancy",
        "chilled_water_temp",
        "Building Load (RT)",
        "Chiller Energy Consumption (kWh)",
    ]
    X = df[features]
    y = df["heating_load"]
    #test and train split to improve model performance
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=19)
    #regression model defenition
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=13
    )
    model.fit(Xtrain, Ytrain)

    y_pred = model.predict(Xtest)

    # model metrics
    mae = mean_absolute_error(Ytest, y_pred)
    mse = mean_squared_error(Ytest, y_pred)
    r2  = r2_score(Ytest, y_pred)


    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("R² Score", f"{r2:.2f}")

    return model

model = train_heating_load_model(data)

# ------------------------ Common Variables ------------------------
baseline_COP_value = 3.75
Electricity_Cost_per_kWh = 0.12
#helper function for calculations so the variables dont have to be defined multiple times
def calculations(indoortemp, temp, dewpoint):
    EXTtemp = temp
    es_dew = 6.112 * np.exp((17.62 * dewpoint) / (243.12 + dewpoint))
    es_temp = 6.112 * np.exp((17.62 * EXTtemp) / (243.12 + EXTtemp))
    hum = 100 * es_dew / es_temp
    occ = np.random.randint(0, 10)
    buildingload = 300 + 0.8 * occ + 2.5 * (EXTtemp - indoortemp) + np.random.normal(0, 10)
    chillerenergy = 500 + 0.2 * (buildingload) + np.random.normal(0, 20)
    return hum, occ, buildingload, chillerenergy
#range for CWT temps that are tested to find the best efficiency, the range is hardcoded and cant be changed from the app 
CWTtemps = np.linspace(8, 14, 31)

st.markdown("""
<div class="main-wrapper">
""", unsafe_allow_html=True)

# ------------------------ Simulations ------------------------
if sim_option == "Temperature Range Simulation":
    # Range-based simulation
    all_results = []
    for _ in range(50):
        EXTtemp = np.random.uniform(5, 35)
        dewpoint = np.random.uniform(5, 30)
        indoortemp = np.random.uniform(64, 79)
        # Convert from F to C
        indoortemp = (indoortemp - 32) * 5/9

        hum, occ, buildingload, chillerenergy = calculations(indoortemp, EXTtemp, dewpoint)

        for CWTtemp in CWTtemps:
            PredVars = pd.DataFrame({
                "Desired Indoor Temperature (C)": [indoortemp],
                "external_temp": [EXTtemp],
                "Humidity (%)": [hum],
                "occupancy": [occ],
                "chilled_water_temp": [CWTtemp],
                "Building Load (RT)": [buildingload],
                "Chiller Energy Consumption (kWh)": [chillerenergy]
            })
            #calculate basic heatpump metrics, more are still needed
            predicted_load = model.predict(PredVars)[0]
            optimized_power, eta = heat_pump_efficiency(indoortemp, EXTtemp, CWTtemp, predicted_load)
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
    # Group and find best chilled water temperature for each external temperature
    OptimizedCWT = (
        all_results.groupby("external_temp", as_index=False)
        .apply(lambda g: g.loc[g["optimized_power_per_ton"].idxmin()])
        .reset_index(drop=True)
    )

    st.write("### Best Chilled Water Temp for Each Outside Temperature")
    OptimizedCWT["total_baseline_power"] = OptimizedCWT["baseline_power_per_ton"] * Toninput
    OptimizedCWT["total_optimized_power"] = OptimizedCWT["optimized_power_per_ton"] * Toninput
    OptimizedCWT["total_savings"] = OptimizedCWT["total_baseline_power"] - OptimizedCWT["total_optimized_power"]
    st.dataframe(OptimizedCWT)

    fig1 = px.line(
        OptimizedCWT,
        x="external_temp",
        y="chilled_water_temp",
        markers=True,
        title="Best CWT vs. Outside Temperature"
    )
  
    fig1.update_layout(
        xaxis_title="Outside Temperature (°C)",
        yaxis_title="Best Chilled Water Temperature (°C)"
    )

    fig2 = px.line(
        OptimizedCWT,
        x="external_temp",
        y=["optimized_power_per_ton", "baseline_power_per_ton"],
        markers=True,
        title="Optimized vs. Baseline Power"
    )
    fig2.update_layout(
        xaxis_title="Outside Temperature (°C)",
        yaxis_title="Power (kW per ton)"
    )

    st.markdown("""
    <div class="chart-row">
      <div class="chart-col">
    """, unsafe_allow_html=True)
    st.plotly_chart(fig1)

    st.markdown("""
      </div>
      <div class="chart-col">
    """, unsafe_allow_html=True)
    st.plotly_chart(fig2)

    st.markdown("""
      </div>
    </div>
    """, unsafe_allow_html=True)

elif sim_option == "Specific Temperature Simulation":
    all_results = []
    CWTtemps = np.linspace(4, 30, 31)
    
    hum1, occ1, buildingload, chillerenergy = calculations(indoortemp, temp, dewpoint)

    for x in range(24):
        for CWTtemp in CWTtemps:
            PredVars = pd.DataFrame({
                "Desired Indoor Temperature (C)": [indoortemp],
                "external_temp": [temp],
                "Humidity (%)": [hum],
                "occupancy": [occ],
                "chilled_water_temp": [CWTtemp],
                "Building Load (RT)": [buildingload],
                "Chiller Energy Consumption (kWh)": [chillerenergy]
            })

            predicted_load = model.predict(PredVars)[0]
            optimized_power, eta = heat_pump_efficiency(indoortemp, temp, CWTtemp, predicted_load)
            baseline_power = predicted_load / baseline_COP_value
            efficiency_gain = ((baseline_power - optimized_power) / baseline_power) * 100
            total_savings = baseline_power - optimized_power
            total_savings_dollars_hour = (total_savings * Electricity_Cost_per_kWh) / 24

            all_results.append({
                "Hour of the day": x,
                "external_temp": temp,
                "Occupants": occ,
                "chilled_water_temp": CWTtemp,
                "predicted_load_per_ton (kWh)": predicted_load,
                "optimized_power_per_ton (kWh)": optimized_power,
                "baseline_power_per_ton (kWh)": baseline_power,
                "efficiency_gain": efficiency_gain,
                "Total_savings (kWh)": total_savings,
                "Total_savings_dollars_hour": total_savings_dollars_hour
            })

        occ = np.random.randint(1, 15)

    all_results = pd.DataFrame(all_results)
    all_results = (
        all_results.groupby("Hour of the day", as_index=False)
        .apply(lambda g: g.loc[g["optimized_power_per_ton (kWh)"].idxmin()])
        .reset_index(drop=True)
    )
    

    fig3 = px.line(
        all_results,
        x="Hour of the day",
        y="Total_savings_dollars_hour",
        markers=True,
        title="Daily savings with differing occupants per hour"
    )
    fig3.update_layout(xaxis_title="Hour", yaxis_title="Dollar Savings")
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(all_results)
# close the main-wrapper div
st.markdown("</div>", unsafe_allow_html=True)
