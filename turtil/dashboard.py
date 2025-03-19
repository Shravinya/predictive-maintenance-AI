import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model & scaler
model = joblib.load("predictive_maintenance_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🔧 Predictive Maintenance Dashboard")

# Sidebar for user input
st.sidebar.header("📊 Input Machine Metrics")
metrics = {}

# Define the required features (from train.py)
required_features = ['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric8', 'metric9']

for feature in required_features:
    metrics[feature] = st.sidebar.number_input(f"{feature}", min_value=0.0, step=1.0)

# Convert input to DataFrame
input_data = pd.DataFrame([metrics])

# Normalize input data
input_scaled = scaler.transform(input_data)

# Predict button
if st.sidebar.button("🔮 Predict Failure"):
    prediction = model.predict(input_scaled)
    failure_risk = "⚠ Failure Expected" if prediction[0] == 1 else "✅ No Failure Detected"
    
    st.subheader("Prediction Result")
    st.write(failure_risk)
    
    # Visualization
    st.subheader("📈 Machine Metrics Overview")
    fig, ax = plt.subplots()
    sns.barplot(x=required_features, y=list(metrics.values()), ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Display dataset (optional)
if st.checkbox("Show Sample Data"):
    df = pd.read_csv("your_dataset.csv")  # Update with your dataset
    st.write(df.head())
