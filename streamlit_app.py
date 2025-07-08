import os
import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_BUNDLE_FILE_ID = "1Af_CzpmK23MqA3x0N0u9wQTt9SyHWPul"

# Download fraud_detection_artifacts.pkl from Google Drive if not present
os.makedirs("models", exist_ok=True)
if not os.path.exists("models/fraud_detection_artifacts.pkl"):
    url = f"https://drive.google.com/uc?id={MODEL_BUNDLE_FILE_ID}"
    gdown.download(url, "models/fraud_detection_artifacts.pkl", quiet=False)

# Load all artifacts
bundle = joblib.load("models/fraud_detection_artifacts.pkl")
lr_best = bundle['lr_best']
rf_best = bundle['rf_best']
xgb_best = bundle['xgb_best']
scaler = bundle['scaler']
le = bundle['le']

# Transaction type mapping
type_label_mapping = {
    'PAYMENT': 0,
    'TRANSFER': 1,
    'CASH_OUT': 2,
    'DEBIT': 3,
    'CASH_IN': 4
}
types = list(type_label_mapping.keys())

st.set_page_config(
    page_title="Fraud Detection",
    layout="centered"
)

st.title("💳 Fraud Detection")

st.markdown("""
<p style="font-size:16px">
Built with <b>Scikit-learn</b>, <b>XGBoost</b> & <b>Streamlit</b> — by Kuldii Project
</p>

<p style="font-size:14px">
This app predicts the probability of a financial transaction being fraudulent based on PaySim simulation data.<br>
✅ Select a machine learning model<br>
📝 Enter transaction parameters<br>
🔮 Get fraud prediction and probability<br>
📂 <strong>Dataset:</strong> <a href="https://www.kaggle.com/datasets/ealaxi/paysim1" target="_blank">
PaySim Financial Transaction Simulator</a> from Kaggle.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

with st.form("fraud_form"):
    st.subheader("📝 Input Transaction Details")

    col1, col2, col3 = st.columns(3)

    step = col1.number_input(
        "⏱️ Step (Hour of Simulation)",
        min_value=1,
        value=1
    )

    type_ = col2.selectbox(
        "💰 Transaction Type",
        types,
        index=0
    )

    amount = col3.number_input(
        "💵 Amount",
        min_value=0.0,
        value=1000.0
    )

    col4, col5, col6, col7 = st.columns(4)

    oldbalanceOrg = col4.number_input(
        "🏦 Old Balance Origin",
        min_value=0.0,
        value=5000.0
    )

    newbalanceOrig = col5.number_input(
        "🏦 New Balance Origin",
        min_value=0.0,
        value=4000.0
    )

    oldbalanceDest = col6.number_input(
        "🏦 Old Balance Dest",
        min_value=0.0,
        value=1000.0
    )

    newbalanceDest = col7.number_input(
        "🏦 New Balance Dest",
        min_value=0.0,
        value=2000.0
    )

    model_name = st.selectbox(
        "🤖 Select Model",
        ["Logistic Regression", "Random Forest", "XGBoost"],
        index=0
    )

    submitted = st.form_submit_button("🚀 Predict Fraud")

if submitted:
    # Prepare input
    type_code = type_label_mapping[type_]
    input_dict = {
        'step': [step],
        'type': [type_code],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    }
    input_df = pd.DataFrame(input_dict)

    # Encode categorical variable
    input_df['type'] = le.transform(input_df['type'])

    # Scale numerical variables
    input_scaled = scaler.transform(input_df)

    # Choose model
    model = {
        'Logistic Regression': lr_best,
        'Random Forest': rf_best,
        'XGBoost': xgb_best
    }[model_name]

    proba = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    pred_text = '🛑 **Fraud**' if pred == 1 else '✅ **Not Fraud**'

    st.markdown(f"""
    ### 🔍 Prediction Result

    **Model:** {model_name}  
    **Fraud Probability:** `{proba:.4f}`  
    **Prediction:** {pred_text}
    """)