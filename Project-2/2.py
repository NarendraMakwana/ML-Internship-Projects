import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Title
st.title("🚗 Vehicle Price Prediction App")

# Upload CSV
uploaded_file = "dataset.csv"

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Data")
    st.write(data.head())

    # Drop rows with null values
    data.dropna(inplace=True)

    # Select features
    features = ['year', 'mileage', 'cylinders', 'fuel', 'transmission', 'body', 'doors', 'drivetrain', 'make', 'model']
    target = 'price'

    # Filter usable features
    data = data[features + [target]]

    # Encode categorical variables
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Split data
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"✅ Model trained. Test MSE: {mse:.2f}")

    # Input section
    st.subheader("🔍 Predict Price")

    year = st.slider("Year", 1990, 2025, 2015)
    mileage = st.number_input("Mileage", 0, 300000, 50000)
    cylinders = st.selectbox("Cylinders", sorted(data['cylinders'].unique()))
    fuel = st.selectbox("Fuel Type", sorted(data['fuel'].unique()))
    transmission = st.selectbox("Transmission", sorted(data['transmission'].unique()))
    body = st.selectbox("Body Style", sorted(data['body'].unique()))
    doors = st.selectbox("Number of Doors", sorted(data['doors'].unique()))
    drivetrain = st.selectbox("Drivetrain", sorted(data['drivetrain'].unique()))
    make = st.selectbox("Make", sorted(data['make'].unique()))
    model_name = st.selectbox("Model", sorted(data['model'].unique()))

    input_df = pd.DataFrame({
        'year': [year],
        'mileage': [mileage],
        'cylinders': [cylinders],
        'fuel': [fuel],
        'transmission': [transmission],
        'body': [body],
        'doors': [doors],
        'drivetrain': [drivetrain],
        'make': [make],
        'model': [model_name]
    })

    # Encode input
    for col in input_df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        le.fit(data[col])
        input_df[col] = le.transform(input_df[col])

    if st.button("Predict"):
        pred_price = model.predict(input_df)[0]
        st.success(f"Estimated Vehicle Price: ${pred_price:,.2f}")
