import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----- Page Configuration -----
st.set_page_config(
    page_title="Logistics Risk Mitigation Dashboard",
    page_icon="🚚",
    layout="wide"
)

# ----- Load Models -----
@st.cache_resource
def load_models():
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('xgboost_model.joblib')
    return preprocessor, model

try:
    preprocessor, model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ----- Dashboard Header -----
st.title("🚚 Predictive Analytics for Project Risk Mitigation")
st.markdown("""
Welcome to the **Logistics Risk Mitigation Dashboard**. Enter the real-time operational metrics and IoT sensor data below to predict the probability of logistics delays and receive proactive resource buffer recommendations.
""")

st.divider()

# ----- Input Features Layout -----
st.header("1. Input Operational & Sensor Data")

# Create columns for better grouping
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Temporal Data")
    # hour, dayofweek, month
    Hour = st.slider("Hour of Day (0-23)", 0, 23, 12, help="0 is Midnight, 23 is 11 PM.")
    DayOfWeek = st.selectbox("Day of Week", options=range(7), format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])
    Month = st.slider("Month (1-12)", 1, 12, 6)
    
    st.subheader("Categorical Data")
    Asset_ID = st.text_input("Asset ID", value="ASST-100", help="Unique identifier for the asset/vehicle.")
    Traffic_Status = st.selectbox("Traffic Status", options=["Clear", "Moderate", "Heavy", "Congested"], index=0)

with col2:
    st.subheader("Environmental IoT Sensors")
    Temperature = st.number_input("Temperature (°C)", value=25.0)
    Humidity = st.number_input("Humidity (%)", value=50.0)
    
    st.subheader("Operational Metrics")
    Waiting_Time = st.number_input("Waiting Time (minutes)", value=15.0)
    Asset_Utilization = st.number_input("Asset Utilization (%)", min_value=0.0, max_value=100.0, value=75.0)

with col3:
    st.subheader("Supply Chain KPIs")
    Inventory_Level = st.number_input("Inventory Level", value=1000.0)
    Demand_Forecast = st.number_input("Demand Forecast", value=500.0)
    
    st.subheader("User Metrics")
    User_Transaction_Amount = st.number_input("Transaction Amount ($)", value=150.0)
    User_Purchase_Frequency = st.number_input("Purchase Frequency", value=5.0)

st.divider()

# ----- Prediction Logic -----
st.header("2. Risk Analysis")

if st.button("Predict Logistics Delay Risk", type="primary"):
    # Create an input dataframe
    input_data = pd.DataFrame([{
        "Inventory_Level": Inventory_Level,
        "Temperature": Temperature,
        "Humidity": Humidity,
        "Waiting_Time": Waiting_Time,
        "User_Transaction_Amount": User_Transaction_Amount,
        "User_Purchase_Frequency": User_Purchase_Frequency,
        "Asset_Utilization": Asset_Utilization,
        "Demand_Forecast": Demand_Forecast,
        "Hour": Hour,
        "DayOfWeek": DayOfWeek,
        "Month": Month,
        "Asset_ID": Asset_ID,
        "Traffic_Status": Traffic_Status
    }])
    
    try:
        # Preprocess
        X_processed = preprocessor.transform(input_data)
        
        # Predict probability
        proba = model.predict_proba(X_processed)[0]
        delay_prob = proba[1] * 100  # probability of class 1 (Delayed)
        
        # Prediction Output
        st.subheader("Risk Probability Score")
        
        # Display progressive metrics depending on risk severity
        if delay_prob < 30:
            st.metric(label="Delay Probability", value=f"{delay_prob:.2f}%", delta="Low Risk", delta_color="normal")
            st.success("**Resource Buffer Recommendation**: Minimal risk. No immediate action required. Maintain standard monitoring schedules.")
            
        elif delay_prob < 70:
            st.metric(label="Delay Probability", value=f"{delay_prob:.2f}%", delta="Medium Risk", delta_color="off")
            st.warning("**Resource Buffer Recommendation**: Moderate risk detected. Consider reviewing resource allocation, preparing backup routes, and alerting stakeholders.")
            
        else:
            st.metric(label="Delay Probability", value=f"{delay_prob:.2f}%", delta="High Risk", delta_color="inverse")
            st.error("**Resource Buffer Recommendation**: Critical risk limit exceeded! Implement resource buffers immediately, initiate proactive contingency plans, and reroute if possible.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
