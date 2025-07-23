import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# -------------------------------
# Load model and training data
# -------------------------------
model = joblib.load("xgb_model_rhob.pkl")
df = pd.read_csv('Data/force2020_data_unsupervised_learning.csv')

# Define the exact feature order used during model training
model_feature_order = ['GR', 'DEPTH_MD', 'NPHI', 'PEF', 'DTC']

# Drop rows with missing values (for LIME training data)
X_train = df[model_feature_order].dropna()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🛢️ Geological Logging: RHOB Prediction with XGBoost")
st.markdown("Enter feature values below or click 'Use Sample Data'.")

# Sample data (can be replaced or expanded)
sample_data = {
    'DTC': 0.75,
    'NPHI': 0.77,
    'GR': -0.85,
    'DEPTH_MD': 2018.02,
    'PEF': 1.47
}

# Input form
with st.form("input_form"):
    DTC = st.number_input("DTC", value=sample_data['DTC'])
    NPHI = st.number_input("NPHI", value=sample_data['NPHI'])
    GR = st.number_input("GR", value=sample_data['GR'])
    DEPTH_MD = st.number_input("DEPTH_MD", value=sample_data['DEPTH_MD'])
    PEF = st.number_input("PEF", value=sample_data['PEF'])

    submitted = st.form_submit_button("🔍 Predict")

# Prepare input and reorder columns to match model
input_data = pd.DataFrame([{
    'DTC': DTC,
    'NPHI': NPHI,
    'GR': GR,
    'DEPTH_MD': DEPTH_MD,
    'PEF': PEF
}])[model_feature_order]

# -------------------------------
# Prediction + LIME explanation
# -------------------------------
if submitted:
    prediction = model.predict(input_data)[0]
    st.success(f"**Predicted RHOB:** {prediction:.4f}")

    st.subheader("📊 LIME Explanation")
    with st.spinner("Generating explanation..."):
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=model_feature_order,
            mode='regression'
        )
        explanation = explainer.explain_instance(
            data_row=input_data.iloc[0].values,
            predict_fn=model.predict
        )

        # Show explanation as text
        for feature, weight in explanation.as_list():
            st.write(f"**{feature}**: {weight:.4f}")

        # Plot explanation
        fig = explanation.as_pyplot_figure()
        st.pyplot(fig)
