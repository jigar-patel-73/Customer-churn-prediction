import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 Project Info")
st.sidebar.write("**Model Used:** Logistic Regression")
st.sidebar.write("**Use Case:** Predict if a customer may churn.")
st.sidebar.write("**Tech Stack:** Python, pandas, scikit-learn, Streamlit")
st.sidebar.write("---")
st.sidebar.write("Built by **Jigar Patel**")

# -----------------------------
# Main Heading
# -----------------------------
st.title("📊 Customer Churn Intelligence")
st.subheader("Predict Customer Retention Risk using Machine Learning")
st.caption("Interactive Churn Prediction Dashboard")

st.write("---")

# -----------------------------
# Inputs
# -----------------------------
st.header("🧾 Enter Customer Details")

senior = st.selectbox("Is Senior Citizen?", [0, 1])

tenure = st.slider(
    "Customer Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.slider(
    "Monthly Charges",
    min_value=0.0,
    max_value=150.0,
    value=50.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# -----------------------------
# Prepare Input Data
# -----------------------------
input_data = pd.DataFrame({
    "SeniorCitizen": [senior],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "Contract": [contract],
    "InternetService": [internet],
    "PaymentMethod": [payment]
})

input_data = pd.get_dummies(input_data, drop_first=True)

for col in model_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model_columns]

# -----------------------------
# Prediction
# -----------------------------
st.write("")

if st.button("🔍 Analyze Churn Risk"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write("---")
    st.subheader("📈 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Churn Risk Detected ({probability:.2%})")
        st.info("Suggested Action: Offer retention discount, loyalty plan, or support outreach.")

    else:
        st.success(f"✅ Strong Retention Probability ({1 - probability:.2%})")
        st.info("Suggested Action: Maintain engagement and upsell premium services.")

# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.caption("Customer Churn Intelligence | Portfolio Project by Jigar Patel")