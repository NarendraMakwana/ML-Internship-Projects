import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Title
st.title("❤️ Heart Disease Prediction App")
st.subheader("Using Machine Learning and Patient Data")

# Load dataset
data = pd.read_csv("dataset.csv")
st.write("### Sample Data", data.head())

# Features and Target
X = data.iloc[:,:-1]
y = data["target"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

# Sidebar for user input
st.sidebar.header("Enter Patient Data")

def user_input():
    age = st.sidebar.slider("Age", 29, 77, 50)
    sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.sidebar.selectbox("Chest Pain Type", [1, 2, 3, 4])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 120, 600, 240)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox("Slope of ST Segment", [1, 2, 3])

    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope]],
                              columns=X.columns)
    return input_data

input_df = user_input()

# Prediction
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]
prediction_proba = model.predict_proba(scaled_input)[0]

# Output
st.write("### Patient Data", input_df)

if prediction == 1:
    st.error("⚠️ The patient is likely to have Heart Disease.")
else:
    st.success("✅ The patient is unlikely to have Heart Disease.")

st.write(f"Prediction Confidence: {prediction_proba[prediction] * 100:.2f}%")
