import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.title("🩺 Liver Cirrhosis Stage Detection")



data = pd.read_csv("liver_cirrhosis.csv")
st.subheader("Preview of Dataset")
st.write(data.head())


label_cols = ["Status", "Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]
le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

st.write("Categorical Columns Encoded.")


X = data.drop("Stage", axis=1)
y = data["Stage"]

    # Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

    # Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

    # Metrics
acc = accuracy_score(y_test, y_pred)
st.subheader("Model Accuracy")
st.success(f"{acc:.2f}")


    # User Input for Prediction
st.subheader("Predict Stage from Patient Data")
input_data = {}
for col in X.columns:
    if col in label_cols:
        input_data[col] = st.selectbox(f"{col}:", le_dict[col].classes_)
    else:
        input_data[col] = st.number_input(f"{col}:", min_value=0.0)

if st.button("Predict Stage"):
        # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

        # Encode categorical input
    for col in label_cols:
        input_df[col] = le_dict[col].transform([input_df[col][0]])

        # Scale numeric input
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    st.success(f"Predicted Cirrhosis Stage: {int(prediction[0])}")
