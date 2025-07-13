import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler

# --- Load Models and Scaler ---
@st.cache_resource
def load_xgb_model():
    model = xgb.Booster()
    model.load_model("models/xgboost_fraud.json")
    return model

@st.cache_resource
def load_autoencoder():
    class Autoencoder(nn.Module):
        def __init__(self, input_dim=30, latent_dim=20):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 24), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(24, 16), nn.ReLU(),
                nn.Linear(16, latent_dim), nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16), nn.ReLU(),
                nn.Linear(16, 24), nn.ReLU(),
                nn.Linear(24, input_dim)
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = Autoencoder()
    model.load_state_dict(torch.load("models/autoencoder_fraud.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl")

# --- UI ---
st.title("💳 Credit Card Fraud Detection")
st.sidebar.title("Model Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["XGBoost", "Autoencoder"])

uploaded_file = st.file_uploader("Upload CSV file with transactions", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'Class' in df.columns:
        df.drop(columns=['Class'], inplace=True)
    scaler = load_scaler()
    data_scaled = scaler.transform(df)

    if model_choice == "XGBoost":
        st.subheader("🔍 Using XGBoost (threshold = 0.3)")
        model = load_xgb_model()
        dmatrix = xgb.DMatrix(data_scaled)
        probs = model.predict(dmatrix)
        preds = (probs > 0.3).astype(int)

    else:
        st.subheader("🔍 Using Autoencoder (quantile = 0.99)")
        model = load_autoencoder()
        inputs = torch.FloatTensor(data_scaled)
        with torch.no_grad():
            recon = model(inputs)
            errors = torch.mean((recon - inputs) ** 2, dim=1).numpy()
        threshold = np.quantile(errors, 0.99)
        preds = (errors > threshold).astype(int)

    df_results = df.copy()
    df_results['Fraud_Prediction'] = preds

    st.success(f"✅ Fraudulent Transactions Detected: {preds.sum()} / {len(preds)}")
    st.dataframe(df_results.head())

    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results", csv, "fraud_predictions.csv", "text/csv")

else:
    st.info("👆 Upload a CSV file to begin")
