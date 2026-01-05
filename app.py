import streamlit as st
import pandas as pd
import joblib

# =====================================
# Load trained pipeline model
# =====================================
@st.cache_resource
def load_model():
    return joblib.load("loan_default_model.pkl")

model = load_model()


# =====================================
# Streamlit UI
# =====================================
st.title("Loan Default Risk Prediction")
st.write("Enter applicant details to estimate default risk.")


# =====================================
# Input Form
# =====================================
with st.form("loan_form"):
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Annual Income", min_value=0, value=50000)
    person_home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )
    person_emp_length = st.number_input(
        "Employment Length (years)", min_value=0, max_value=60, value=5
    )
    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION",
        ]
    )
    loan_grade = st.selectbox(
        "Loan Grade", ["A", "B", "C", "D", "E", "F", "G"]
    )
    loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.5)
    loan_percent_income = st.number_input(
        "Loan Percent of Income", min_value=0.0, max_value=1.0, value=0.25
    )
    cb_person_default_on_file = st.selectbox(
        "Previous Default on File", ["Y", "N"]
    )
    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (years)", min_value=0, max_value=50, value=8
    )

    submitted = st.form_submit_button("Predict Risk")


# =====================================
# Prediction
# =====================================
if submitted:
    # IMPORTANT:
    # - Column names must match training
    # - loan_status MUST NOT be included
    # - Order does NOT matter (pipeline uses column names)

    new_applicant = pd.DataFrame(
        [{
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
        }]
    )

    # Predict probability of default
    risk_prob = model.predict_proba(new_applicant)[0, 1]

    # Business threshold
    THRESHOLD = 0.6
    decision = "REJECT LOAN" if risk_prob >= THRESHOLD else "APPROVE LOAN"

    # Output
    st.subheader("Result")
    st.write(f"**Default Risk Probability:** {risk_prob:.2f}")
    st.write(f"**Decision:** {decision}")

    if decision == "REJECT LOAN":
        st.error("High risk of default detected.")
    else:
        st.success("Applicant is likely to repay the loan.")
