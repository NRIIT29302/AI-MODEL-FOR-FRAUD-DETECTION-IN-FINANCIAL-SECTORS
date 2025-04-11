import streamlit as st
import numpy as np
import pandas as pd
import joblib # Import joblib
from sklearn.linear_model import LogisticRegression # Keep if you load LR
from sklearn.ensemble import RandomForestClassifier # Add if you load RF

try:
    # Choose the model file you saved in Step 1
    model = joblib.load('fraud_detection_random_forest_model.joblib')

    feature_columns = joblib.load('feature_columns.joblib')
    EXPECTED_FEATURES = len(feature_columns)
    model_loaded = True
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'fraud_detection_logistic_model.joblib' (or RF model) and 'feature_columns.joblib' are in the same directory as app.py.")
    model_loaded = False
except Exception as e:
    st.error(f"Error loading model or features: {e}")
    model_loaded = False


# --- Streamlit App ---
st.title("Credit Card Fraud Detection Model")

if model_loaded:
    st.write("Using pre-trained model.") # Indicate the model is loaded
    st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

    # Create input fields for user to enter feature values.
    st.write(f"Provide all {EXPECTED_FEATURES} feature values separated by commas (e.g., -1.35, ..., 0.73). These correspond to the columns V1 to Amount in the dataset.")
    # Display feature names if helpful (optional)
    # st.write(f"Feature order: {', '.join(feature_columns)}")

    input_df = st.text_input("Input All features", key="feature_input")

    # Submit Button
    submit = st.button("Submit")

    if submit:
        if input_df: # Check if not empty
            try:
                # Get input feature values and convert to float64
                features = np.asarray(input_df.split(','), dtype=np.float64)

                # Check if the number of features is correct
                if features.shape[0] != EXPECTED_FEATURES:
                    st.error(f"Incorrect number of features. Please provide {EXPECTED_FEATURES} features.")
                else:
                    # Make prediction USING THE LOADED MODEL
                    prediction = model.predict(features.reshape(1, -1))

                    # Display result
                    if prediction[0] == 0:
                        st.success("Legitimate Transaction") # Use success/error for better visual cues
                    else:
                        st.error("Fraudulent Transaction")
            except ValueError:
                st.error("Invalid input. Please enter numeric values separated by commas.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter the feature values.")

else:
    st.info("Could not load the prediction model. Application cannot proceed.")