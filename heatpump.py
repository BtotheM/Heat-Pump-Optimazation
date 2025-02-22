import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
LOGO = "https://github.com/BtotheM/Heat-Pump-Optimazation/blob/main/Teamlogo.png?raw=true"
DATALINK = "https://github.com/BtotheM/Heat-Pump-Optimazation/blob/main/augmented_dataset.csv?raw=true"
# ---- Inject Custom CSS ----
st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

# ------------------------ Title ------------------------
st.title("Random Forest Regression Model: Optimizing Water-to-Water Set Points")

# ------------------------ Sidebar: Inputs & Navigation ------------------------
with st.sidebar:
    st.image(LOGO, width=400)
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
        # We'll interpret this as the average daily outside temperature
        temp = st.number_input("Outside Temperature (°C)", value=20.0, min_value=-50.0, max_value=100.0)
        indoortemp = st.number_input("Desired Indoor Temperature (°C)", value=22.0, step=1.0)
        dewpoint = st.number_input("Dew Point (°C)", value=25.0, min_value=-50.0, max_value=100.0)
        hum = st.number_input("Humidity (%)", value=50, min_value=0, max_value=100)
        occ = st.number_input("Occupants in Building:", value=1, min_value=1, max_value=15)
        building_area_input = st.number_input("Building Area (sq ft):", value=2000, min_value=500, max_value=100000)
    else:
        st.write("**Range Simulation** will use random data.")

# ------------------------ Heat Pump Efficiency ------------------------
def heat_pump_efficiency(indoortemp, external_temp, chilled_water_temp, heating_demand):
    """
    Calculates the power input using a Carnot COP model with an adaptive efficiency factor.
    T_hot is fixed at 313.15 K (40°C) or user-supplied indoors, T_cold from chilled water.
    """
    T_hot = indoortemp + 273.15
    T_cold = max(chilled_water_temp + 273.15, 273.16)
    temperature_difference = T_hot - T_cold
    eta = np.clip(0.85 - 0.003 * max(temperature_difference - 5, 0), 0.55, 0.85)
    COP_carnot = T_hot / (T_hot - T_cold) if (T_hot - T_cold) > 0 else 1
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
    
    # Building area
    building_area = np.random.normal(2000, 500, n)
    building_area = np.clip(building_area, 500, 5000)
    data["Building Area (sq ft)"] = building_area
    
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
        + 0.0005 * (data["Building Area (sq ft)"] - 2000)
        + noise
    )
    data["heating_load"] = data["heating_load"].abs()
    return data

uploaded_file = DATALINK


data = pd.read_csv(uploaded_file)
data = CreateSyntheticCols(data)
st.dataframe(data.head(100000))

# Train Random Forest Model
def train_heating_load_model(df):
    features = [
        "Desired Indoor Temperature (C)",
        "external_temp",
        "Humidity (%)",
        "occupancy",
        "chilled_water_temp",
        "Building Load (RT)",
        "Chiller Energy Consumption (kWh)",
        "Building Area (sq ft)",
    ]
    X = df[features]
    y = df["heating_load"]
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=19)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=13
    )
    model.fit(Xtrain, Ytrain)

    y_pred = model.predict(Xtest)

    mae = mean_absolute_error(Ytest, y_pred)
    r2  = r2_score(Ytest, y_pred)
    st.write("ML Model Metrics: ")
    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae:.2f}", border = True)
    col2.metric("R² Score", f"{r2:.4f}", border = True)

    return model

model = train_heating_load_model(data)

baseline_COP_value = 3.75
Electricity_Cost_per_kWh = 0.12

def calculations(indoortemp_c, ext_temp_c, dewpoint_c, building_area=None):
    
    # For humidity
    es_dew = 6.112 * np.exp((17.62 * dewpoint_c) / (243.12 + dewpoint_c))
    es_temp = 6.112 * np.exp((17.62 * ext_temp_c) / (243.12 + ext_temp_c))
    hum = 100 * es_dew / es_temp

    # random occupancy if none is specified hour by hour
    occ = np.random.randint(0, 10)

    if building_area is None:
        b_area = np.random.normal(2000, 500)
        b_area = np.clip(b_area, 500, 5000)
    else:
        b_area = building_area

    buildingload = (
        300
        + 0.8 * occ
        + 2.5 * (ext_temp_c - indoortemp_c)
        + 0.0005 * (b_area - 2000)
        + np.random.normal(0, 10)
    )
    chillerenergy = 500 + 0.2 * buildingload + np.random.normal(0, 20)
    return hum, occ, buildingload, chillerenergy, b_area

CWTtemps = np.linspace(8, 14, 31)

st.markdown("""<div class="main-wrapper">""", unsafe_allow_html=True)

# SIms
if sim_option == "Temperature Range Simulation":
   
   
    all_results = []
    for _ in range(50):
        EXTtemp = np.random.uniform(5, 35)
        dewpoint = np.random.uniform(5, 30)
        indoortemp_f = np.random.uniform(64, 79)
        indoortemp_c = (indoortemp_f - 32) * 5/9

        hum_val, occ_val, buildingload, chillerenergy, b_area = calculations(
            indoortemp_c, EXTtemp, dewpoint, building_area=None
        )

        for CWTtemp in CWTtemps:
            PredVars = pd.DataFrame({
                "Desired Indoor Temperature (C)": [indoortemp_c],
                "external_temp": [EXTtemp],
                "Humidity (%)": [hum_val],
                "occupancy": [occ_val],
                "chilled_water_temp": [CWTtemp],
                "Building Load (RT)": [buildingload],
                "Chiller Energy Consumption (kWh)": [chillerenergy],
                "Building Area (sq ft)": [b_area],
            })

            predicted_load = model.predict(PredVars)[0]
            optimized_power, eta = heat_pump_efficiency(indoortemp_c, EXTtemp, CWTtemp, predicted_load)
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

    st.markdown("""<div class="chart-row"><div class="chart-col">""", unsafe_allow_html=True)
    st.plotly_chart(fig1)

    st.markdown("""</div><div class="chart-col">""", unsafe_allow_html=True)
    st.plotly_chart(fig2)

    st.markdown("""</div></div>""", unsafe_allow_html=True)


elif sim_option == "Specific Temperature Simulation":
    # Day/Night cycle simulation
    all_results = []
    CWTtemps = np.linspace(4, 30, 31)
    amplitude = 5.0  
    b_area = building_area_input

    for hour in range(24):
        # Hourly outside temp with day/night cycle
        # Removing '- 6' means hour=0 => EXTtemphour = temp (no shift).
        EXTtemphour = temp + amplitude * np.sin((2*np.pi/24) * (hour - 6))

        # Occupants change every hour
        occ_hour = np.random.randint(1, 15)
        es_dew = 6.112 * np.exp((17.62 * dewpoint) / (243.12 + dewpoint))
        es_temp = 6.112 * np.exp((17.62 * EXTtemphour) / (243.12 + EXTtemphour))
        hum_calc = 100 * es_dew / es_temp
        buildingload = (300 + 0.8 * occ_hour + 2.5 * (EXTtemphour - indoortemp) + 0.0005 * (b_area - 2000) + np.random.normal(0, 10))
        chillerenergy = 500 + 0.2 * buildingload + np.random.normal(0, 20)

        for CWTtemp in CWTtemps:
            PredVars = pd.DataFrame({
                "Desired Indoor Temperature (C)": [indoortemp],
                "external_temp": [EXTtemphour],
                "Humidity (%)": [hum_calc],
                "occupancy": [occ_hour],
                "chilled_water_temp": [CWTtemp],
                "Building Load (RT)": [buildingload],
                "Chiller Energy Consumption (kWh)": [chillerenergy],
                "Building Area (sq ft)": [b_area],
            })

            predicted_load = model.predict(PredVars)[0]
            optimized_power, eta = heat_pump_efficiency(indoortemp, EXTtemphour, CWTtemp, predicted_load)
            baseline_power = predicted_load / baseline_COP_value
            efficiency_gain = ((baseline_power - optimized_power) / baseline_power) * 100

            total_savings = baseline_power - optimized_power
            total_savings_dollars_hour = total_savings * Electricity_Cost_per_kWh / 24

            all_results.append({
                "Hour": hour,
                "OutsideTemp(°C)": EXTtemphour,
                "Occupants": occ_hour,
                "ChilledWaterTemp(°C)": CWTtemp,
                "predicted_load_per_ton(kWh)": predicted_load,
                "optimized_power_per_ton(kWh)": optimized_power,
                "baseline_power_per_ton(kWh)": baseline_power,
                "efficiency_gain(%)": efficiency_gain,
                "Hourly_savings(kWh)": total_savings,
                "Hourly_savings_dollars": total_savings_dollars_hour
            })

    results_df = pd.DataFrame(all_results)

    # For each hour, find the minimal optimized_power
    best_per_hour = (
        results_df.groupby("Hour", as_index=False)
        .apply(lambda g: g.loc[g["optimized_power_per_ton(kWh)"].idxmin()])
        .reset_index(drop=True)
    )

    st.write("### Hourly Best Setpoints & Savings")
    st.dataframe(best_per_hour)

    fig_temp = px.line(
        best_per_hour,
        x="Hour",
        y="ChilledWaterTemp(°C)",
        title="Optimal Chilled Water Temp(°C) Set Point vs. Hour (Day/Night Cycle)",
        markers=True,
    )

    # Plot best savings
    fig_savings = px.line(
        best_per_hour,
        x="Hour",
        y="Hourly_savings_dollars",
        title="Hourly Savings (USD) with Best Set Points",
        markers=True,
    )

    fig2 = px.scatter(
        best_per_hour,
        x="OutsideTemp(°C)",
        y=["optimized_power_per_ton(kWh)", "baseline_power_per_ton(kWh)"],
        title="Optimized vs. Baseline Power input"
    )
    fig2.update_layout(
        xaxis_title="Outside Temperature (°C)",
        yaxis_title="Power (kW per ton)"
    )

    st.plotly_chart(fig_temp, use_container_width=True)
    st.plotly_chart(fig_savings, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    daily_dollar_savings = best_per_hour["Hourly_savings_dollars"].sum()
avg_baseline_power = best_per_hour["baseline_power_per_ton(kWh)"].mean()
avg_optimized_power = best_per_hour["optimized_power_per_ton(kWh)"].mean()
avg_efficiency_gain = best_per_hour["efficiency_gain(%)"].mean()

col1, col2  = st.columns(2)
col3, col4 = st.columns(2)
col1.metric("Daily Dollar Savings", f"${daily_dollar_savings:,.2f}", border = True)
col2.metric("Avg Baseline Power (kW/ton)", f"{avg_baseline_power:.2f}", border = True)
col3.metric("Avg Optimized Power (kW/ton)", f"{avg_optimized_power:.2f}", border = True)
col4.metric("Avg Efficiency Gain (%)", f"{avg_efficiency_gain:.2f}", border = True)
   

st.markdown("</div>", unsafe_allow_html=True)
