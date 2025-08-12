import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv("dataset.csv")

# Prepare features and target
x = data.iloc[:, :-1]
y = data["price_range"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train model
rf = RandomForestClassifier(n_estimators=50)
rf.fit(x_train_scaled, y_train)

# Streamlit app
st.title("📱 Mobile Price Prediction")

# Input fields
battery_power = st.number_input("Battery Power (mAh)", 500, 2000, step=1)
blue = st.selectbox("Bluetooth", [0, 1])
clock_speed = st.number_input("Clock Speed (GHz)", 0.5, 3.0, step=0.1)
dual_sim = st.selectbox("Dual SIM", [0, 1])
fc = st.number_input("Front Camera (MP)", 0, 20)
four_g = st.selectbox("4G Support", [0, 1])
int_memory = st.number_input("Internal Memory (GB)", 2, 128)
m_deep = st.number_input("Mobile Depth (cm)", 0.1, 1.0, step=0.01)
mobile_wt = st.number_input("Weight (grams)", 80, 250)
n_cores = st.number_input("Number of Cores", 1, 8)
pc = st.number_input("Primary Camera (MP)", 2, 20)
px_height = st.number_input("Pixel Height", 0, 2000)
px_width = st.number_input("Pixel Width", 0, 2000)
ram = st.number_input("RAM (MB)", 256, 4000)
sc_h = st.number_input("Screen Height (cm)", 5, 20)
sc_w = st.number_input("Screen Width (cm)", 0, 20)
talk_time = st.number_input("Talk Time (hours)", 2, 24)
three_g = st.selectbox("3G Support", [0, 1])
touch_screen = st.selectbox("Touch Screen", [0, 1])
wifi = st.selectbox("WiFi Support", [0, 1])

# Predict button
if st.button("Predict"):
    user_input = [[
        battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory,
        m_deep, mobile_wt, n_cores, pc, px_height, px_width, ram,
        sc_h, sc_w, talk_time, three_g, touch_screen, wifi
    ]]
    user_input_scaled = scaler.transform(user_input)
    prediction = rf.predict(user_input_scaled)[0]
    st.success(f"Predicted Price Range: {prediction}")
