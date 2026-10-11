import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Page Title
st.set_page_config(page_title="Credit Scoring System", layout="centered")
st.title("💳 Credit Scoring & Risk Prediction Model")
st.write("Enter applicant details below to evaluate creditworthiness.")

# 1. Train Model (Cached for performance)
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 1000
    data = {
        'income': np.random.randint(20000, 120000, size=n_samples),
        'total_debt': np.random.randint(1000, 50000, size=n_samples),
        'payment_history_score': np.random.randint(1, 10, size=n_samples),
        'credit_age_years': np.random.randint(1, 15, size=n_samples)
    }
    df = pd.DataFrame(data)
    df['creditworthiness'] = np.where(
        (df['income'] > 50000) & (df['payment_history_score'] > 5) & (df['total_debt'] < 30000), 1, 0
    )
    
    X = df[['income', 'total_debt', 'payment_history_score', 'credit_age_years']]
    y = df['creditworthiness']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

model, scaler = train_model()

# 2. User Input Form
st.sidebar.header("Applicant Financial Data")
income = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=60000, step=1000)
total_debt = st.sidebar.number_input("Total Debt ($)", min_value=0, max_value=100000, value=15000, step=500)
payment_history = st.sidebar.slider("Payment History Score (1 = Poor, 10 = Excellent)", 1, 10, 7)
credit_age = st.sidebar.slider("Credit History (Years)", 1, 30, 5)

# 3. Prediction Button
if st.button("Predict Creditworthiness"):
    user_data = np.array([[income, total_debt, payment_history, credit_age]])
    user_data_scaled = scaler.transform(user_data)
    
    prediction = model.predict(user_data_scaled)[0]
    prob = model.predict_proba(user_data_scaled)[0][1]
    
    st.markdown("---")
    if prediction == 1:
        st.success(f"🎉 **Approved**: Good Credit Standing! (Confidence: {prob * 100:.1f}%)")
    else:
        st.error(f"⚠️ **High Risk**: Low Credit Score / Reject. (Confidence: {(1 - prob) * 100:.1f}%)")
        