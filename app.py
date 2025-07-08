import joblib
import numpy as np
import gradio as gr
import pandas as pd

# Load all artifacts from single file
artifacts = joblib.load('models/fraud_detection_artifacts.pkl')
lr_best = artifacts['lr_best']
rf_best = artifacts['rf_best']
xgb_best = artifacts['xgb_best']
scaler = artifacts['scaler']
le = artifacts['le']

# Transaction type mapping (adjust if needed)
type_label_mapping = {
    'PAYMENT': 0,
    'TRANSFER': 1,
    'CASH_OUT': 2,
    'DEBIT': 3,
    'CASH_IN': 4
}
types = list(type_label_mapping.keys())

# Gradio prediction function
def predict_fraud(step, type_, amount, oldbalanceOrg,
                newbalanceOrig, oldbalanceDest, newbalanceDest, model_name):
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
    input_df['type'] = le.transform(input_df['type'])
    input_scaled = scaler.transform(input_df)
    model = {
        'Logistic Regression': lr_best,
        'Random Forest': rf_best,
        'XGBoost': xgb_best
    }[model_name]
    proba = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]
    result_text = f"""
    🔍 Prediction Result
    \n**Model:** {model_name}  
    **Fraud Probability:** `{proba:.4f}`  
    **Prediction:** {"🛑 **Fraud**" if pred == 1 else "✅ **Not Fraud**"}
    """
    return result_text

def main():
    with gr.Blocks() as demo:
        gr.Markdown("""
            <h2>💳 Fraud Detection</h2>
            <p style='font-size: 16px;'>Built with Scikit-learn, XGBoost & Gradio — by Kuldii Project</p>
            <p style='font-size: 14px;'>This app predicts the probability of a financial transaction being fraudulent based on PaySim simulation data.<br>
            ✅ Select a machine learning model<br>
            📝 Enter transaction parameters<br>
            🔮 Get fraud prediction and probability<br>
            📂 <strong>Dataset:</strong> <a href='https://www.kaggle.com/datasets/ealaxi/paysim1' target='_blank'>PaySim Financial Transaction Simulator</a> from Kaggle.
            </p>
        """)
        gr.Markdown("""
        ℹ️ *PaySim is a financial transaction simulator commonly used for fraud detection.*  
        This application uses several Machine Learning models (Logistic Regression, Random Forest, XGBoost)
        to predict whether a transaction is fraudulent or not.
        """)
        step_in = gr.Number(label="⏱️ Step (Hour of Simulation)", value=1,
                            info="Simulation hour at which the transaction occurs.")
        type_in = gr.Dropdown(choices=types, label="💰 Transaction Type",
                            info="Type of financial transaction.")
        amount_in = gr.Number(label="💵 Amount", value=1000,
                            info="Transaction amount.")
        oldbalanceOrg_in = gr.Number(label="🏦 Old Balance Origin", value=5000,
                                    info="Balance before the transaction in the sender's account.")
        newbalanceOrig_in = gr.Number(label="🏦 New Balance Origin", value=4000,
                                    info="Balance after the transaction in the sender's account.")
        oldbalanceDest_in = gr.Number(label="🏦 Old Balance Dest", value=1000,
                                    info="Balance before the transaction in the receiver's account.")
        newbalanceDest_in = gr.Number(label="🏦 New Balance Dest", value=2000,
                                    info="Balance after the transaction in the receiver's account.")
        model_in = gr.Radio(
            choices=["Logistic Regression", "Random Forest", "XGBoost"],
            label="🤖 Select Model",
            value="Logistic Regression",
            info="Select the Machine Learning model for prediction."
        )
        btn = gr.Button("🚀 Predict Fraud")
        output = gr.Markdown()
        btn.click(
            fn=predict_fraud,
            inputs=[
                step_in, type_in, amount_in,
                oldbalanceOrg_in, newbalanceOrig_in,
                oldbalanceDest_in, newbalanceDest_in, model_in
            ],
            outputs=output
        )
        demo.launch()

if __name__ == "__main__":
    main()
