# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os # Import os module

# --- Configuration ---
MODEL_PATH = 'fraud_detection_random_forest_model.joblib'
FEATURES_PATH = 'feature_columns.joblib'

# --- Helper Functions ---
@st.cache_resource # Cache the loaded model and features for efficiency
def load_model_and_features():
    """Loads the pre-trained model and feature list."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at {MODEL_PATH}")
        st.stop() # Stop execution if model file is missing
    if not os.path.exists(FEATURES_PATH):
        st.error(f"Error: Feature list file not found at {FEATURES_PATH}")
        st.stop() # Stop execution if feature file is missing

    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        st.success("Model and feature list loaded successfully!")
        return model, feature_names
    except Exception as e:
        st.error(f"Error loading model or feature list: {e}")
        st.stop()

# --- Load Model and Features ---
model, feature_names = load_model_and_features()
st.info(f"Model expects {len(feature_names)} features: {', '.join(feature_names)}")

# --- Streamlit App Interface ---
st.title("💳 Credit Card Fraud Detection")
st.write("""
Enter the transaction details below to predict if it's fraudulent or legitimate.
The prediction is based on a pre-trained Random Forest model.
""")

# --- User Input ---
st.header("Transaction Features")
input_data = {}

# Create input fields dynamically based on loaded feature names
# Use columns for better layout if many features
num_columns = 3 # Adjust as needed
cols = st.columns(num_columns)
col_idx = 0

for feature in feature_names:
    with cols[col_idx % num_columns]:
         # Use a default value (e.g., 0.0) and provide a key for each input
        input_data[feature] = st.number_input(
            label=f"Enter {feature}",
            value=0.0, # Sensible default
            format="%.6f", # Adjust format if needed
            key=f"input_{feature}" # Unique key for each widget
        )
    col_idx += 1


# --- Prediction ---
if st.button("🔍 Predict Fraud"):
    try:
        # Create a DataFrame from the input data in the correct order
        input_df = pd.DataFrame([input_data])
        # Ensure columns are in the same order as when the model was trained
        input_df = input_df[feature_names]

        st.write("---")
        st.subheader("Input Data Overview:")
        st.dataframe(input_df)

        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error(f"🚨 Prediction: Fraudulent Transaction")
            st.warning(f"Confidence Score (Fraud): {prediction_proba[0][1]:.4f}")
        else:
            st.success(f"✅ Prediction: Legitimate Transaction")
            st.info(f"Confidence Score (Legit): {prediction_proba[0][0]:.4f}")

        st.write("---")
        st.write("Probability Scores:")
        st.write(f" - Probability of being Legitimate (Class 0): {prediction_proba[0][0]:.4f}")
        st.write(f" - Probability of being Fraudulent (Class 1): {prediction_proba[0][1]:.4f}")

    except KeyError as e:
         st.error(f"Input data is missing a required feature: {e}. Please ensure all feature values are entered.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Optional: Display Footer ---
st.sidebar.info("App developed for fraud detection demonstration.")