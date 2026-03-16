import streamlit as st
import joblib
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="👥",
    layout="centered"
)

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("Scaler.pkl")
    model  = joblib.load("model.pkl")
    return scaler, model

scaler, model = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("👥 Employee Performance Predictor")
st.markdown(
    "Enter an employee's profile below to receive an instant **Performance Score** "
    "prediction (scale: 1 = lowest → 5 = highest)."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Employee Profile")

col1, col2 = st.columns(2)

with col1:
    years = st.number_input(
        "Years at Company",
        min_value=0, max_value=15, value=2,
        help="How many full years has the employee been with the company?"
    )
    salary = st.number_input(
        "Monthly Salary (₹)",
        min_value=1000, max_value=100000, value=5000, step=500,
        help="Employee's gross monthly salary."
    )
    overtime = st.number_input(
        "Overtime Hours (per week)",
        min_value=0, max_value=29, value=10,
        help="Average overtime hours worked per week."
    )

with col2:
    promotions = st.number_input(
        "Number of Promotions",
        min_value=0, max_value=2, value=0,
        help="Total promotions received during tenure."
    )
    satisfaction = st.number_input(
        "Employee Satisfaction Score",
        min_value=1.0, max_value=5.0, value=3.0, step=0.1,
        help="Self-reported satisfaction score (1.0 = very dissatisfied, 5.0 = very satisfied)."
    )

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
# Feature order must match training: Years_At_Company, Monthly_Salary,
#                                    Overtime_Hours, Promotions, Employee_Satisfaction_Score
if st.button("🔍 Predict Performance Score", use_container_width=True):
    features   = np.array([[years, salary, overtime, promotions, satisfaction]])
    scaled     = scaler.transform(features)
    prediction = model.predict(scaled)[0]

    # Score labels
    labels = {
        1: ("🔴", "Needs Improvement",  "Consider a performance improvement plan and closer mentoring."),
        2: ("🟠", "Below Average",       "Identify skill gaps and provide targeted training."),
        3: ("🟡", "Average",             "Meets expectations. Explore stretch assignments to unlock potential."),
        4: ("🟢", "Above Average",       "Strong performer. Consider for leadership development programmes."),
        5: ("🏆", "Outstanding",         "Top performer. Prioritise retention — promotion or salary review recommended."),
    }

    icon, label, advice = labels.get(prediction, ("❓", "Unknown", ""))

    st.balloons()
    st.success(f"**Predicted Performance Score: {prediction} / 5 — {icon} {label}**")

    st.info(f"💡 **HR Recommendation:** {advice}")

    # Show input summary
    with st.expander("📋 Input Summary"):
        st.write({
            "Years at Company":              years,
            "Monthly Salary (₹)":            salary,
            "Overtime Hours / week":         overtime,
            "Promotions":                    promotions,
            "Satisfaction Score":            satisfaction,
        })
else:
    st.info("Fill in the employee's profile above and click **Predict Performance Score** to get a result.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: SVM with GridSearchCV · Dataset: 100,000 synthetic employee records · "
    "Best validation accuracy: 49.8% (KNN) · "
    "Built with Scikit-learn & Streamlit"
)
