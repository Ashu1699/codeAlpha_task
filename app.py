import streamlit as st
import pandas as pd
import joblib
import time

# Model load karo
model = joblib.load("models/best_credit_model.joblib")

# Page Config
st.set_page_config(page_title="Credit Scoring Dashboard", page_icon="💳", layout="wide")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/747/747376.png", width=100)
    st.title("📊 Dashboard Menu")
    st.markdown("### 🔹 Options")
    st.markdown("- Home\n- Prediction\n- About")

# Title with animation
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>💳 Credit Scoring App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Interactive Prediction Dashboard</h3>", unsafe_allow_html=True)

# Creator name in corner
st.markdown(
    """
    <div style='position: fixed; bottom: 10px; right: 15px; 
                color: gray; font-size: 14px;'>
        🚀 Created by <b>Sandeep Singh</b>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# Form inputs in columns
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Annual Income", 10000, 1000000, 50000)
    loan_amount = st.number_input("Loan Amount", 1000, 500000, 10000)

with col2:
    credit_history_years = st.number_input("Credit History (Years)", 0, 40, 5)
    num_open_loans = st.number_input("Number of Open Loans", 0, 20, 2)
    late_payments_12m = st.number_input("Late Payments (12 months)", 0, 10, 0)

with col3:
    education_level = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed", "Student", "Retired"])

# Predict button with progress
if st.button("🚀 Predict"):
    with st.spinner("Model running... Please wait..."):
        time.sleep(2)  # thoda animation feel ke liye

        input_data = pd.DataFrame([{
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "credit_history_years": credit_history_years,
            "num_open_loans": num_open_loans,
            "late_payments_12m": late_payments_12m,
            "education_level": education_level,
            "marital_status": marital_status,
            "employment_status": employment_status
        }])
        
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

    st.success("✅ Prediction Complete!")

    # Result dashboard
    st.write("### 🧾 Results")
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.metric(label="Prediction", value="❌ Default" if prediction == 1 else "✅ No Default")

    with col_res2:
        st.metric(label="Probability of Default", value=f"{prob:.2%}")
