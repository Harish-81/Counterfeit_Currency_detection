# app.py
import streamlit as st
import numpy as np
import pickle

# Load model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# App title
st.title("💵 Fake Currency Detection")
st.markdown("Enter the currency note statistics below to detect if it's Fake or Genuine.")

# Input fields
variance = st.number_input("Variance", value=0.0, format="%.4f")
skew = st.number_input("Skewness", value=0.0, format="%.4f")
curtosis = st.number_input("Curtosis", value=0.0, format="%.4f")
entropy = st.number_input("Entropy", value=0.0, format="%.4f")

# Prediction
if st.button("Predict"):
    input_data = np.array([[variance, skew, curtosis, entropy]])
    result = model.predict(input_data)[0]
    
    if result == 0:
        st.success("✅ The currency is **Genuine**.")
    else:
        st.error("❌ The currency is **Fake**.")
