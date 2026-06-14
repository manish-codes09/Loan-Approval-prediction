import streamlit as st
import pickle
import numpy as np

# Page Config
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="centered"
)

# Load Model
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1 {
    text-align: center;
    color: #1f4e79;
}

.stButton > button {
    width: 100%;
    height: 50px;
    background-color: #28a745;
    color: white;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #218838;
}

div[data-testid="stNumberInput"] {
    padding-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #dbeafe, #f0f9ff);
}
</style>
""", unsafe_allow_html=True)


# Title
st.title("🏦 Loan Approval Prediction System")
st.markdown("### Enter Loan Details")

# Inputs
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=700
    )

    income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=10000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=10000
    )

with col2:
    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.30
    )

    loan_term = st.number_input(
        "Loan Term (Months)",
        min_value=1,
        value=24
    )

st.write("")

# Prediction
if st.button("🔍 Predict Loan Status"):

    data = np.array([[
        credit_score,
        income,
        loan_amount,
        dti_ratio,
        loan_term
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)
    probability = model.predict_proba(data)

    st.write("### Prediction Probability")
    st.write(f"Rejected : {probability[0][0]*100:.2f}%")
    st.write(f"Approved : {probability[0][1]*100:.2f}%")

    if prediction[0] == 1:
        st.success(
            f"✅ Loan Approved ({probability[0][1]*100:.2f}% Confidence)"
        )
        st.balloons()

    else:
        st.error(
            f"❌ Loan Rejected ({probability[0][0]*100:.2f}% Confidence)"
        )


                         
                         
                         