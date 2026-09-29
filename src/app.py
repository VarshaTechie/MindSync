import streamlit as st
import pandas as pd
import sys
import os
import time

# Fix path for utils
sys.path.append(os.path.dirname(__file__))
from utils import predict_risk, load_model_and_encoders

st.set_page_config(page_title="Student Mental Health Survey", page_icon="🧠", layout="centered")

# Initialize session state
if 'risk' not in st.session_state:
    st.session_state.risk = None
    st.session_state.result = None

# Function to interpret risk
def interpret_risk(risk):
    if risk == 'Low':
        return "No significant mental health issues detected. Keep maintaining a healthy lifestyle!"
    elif risk == 'Medium':
        return "Potential mental health issues; consider monitoring your well-being and talking to a counselor if needed."
    elif risk == 'High':
        return "Possible mental health concerns. Please seek professional help from a mental health expert."
    else:
        return "Unable to determine. Please consult a professional."

# 🌟 Animated Risk Bar
def animated_risk_bar(risk):
    fill_values = {"Low": 33, "Medium": 66, "High": 100}
    colors = {"Low": "#81C784", "Medium": "#FFF176", "High": "#E57373"}
    emojis = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

    target_fill = fill_values[risk]
    bar_container = st.empty()  # placeholder for the bar

    for i in range(0, target_fill + 1, 5):
        html_code = f"""
        <div style="font-family:sans-serif; text-align:center; margin:20px 0;">
            <h3>Your Mental Health Status</h3>
            <div style="
                background: #eee;
                border-radius: 15px;
                height: 40px;
                width: 80%;
                margin: 0 auto;
                overflow: hidden;
            ">
                <div style="
                    width: {i}%;
                    height: 100%;
                    background-color: {colors[risk]};
                    border-radius: 15px;
                    transition: width 0.2s ease;
                "></div>
            </div>
            <p style="margin-top:10px; font-size:18px;">{emojis[risk]} <b>{risk} Risk</b></p>
        </div>
        """
        bar_container.markdown(html_code, unsafe_allow_html=True)
        time.sleep(0.05)

# Emoji Feedback
def show_feedback(risk):
    if risk == "Low":
        st.markdown("### 🟢 Low Risk – You’re doing great! Keep healthy habits 💪")
    elif risk == "Medium":
        st.markdown("### 🟡 Medium Risk – Stay mindful 🌱 Take breaks, journal, and talk to friends.")
    elif risk == "High":
        st.markdown("### 🔴 High Risk – It’s okay to not feel okay ❤️ Consider professional help.")

# Title
st.title("🧠 Student Mental Health Survey")
st.header("Answer the following questions to assess your mental wellness")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=30, value=20)
gender = st.selectbox("Gender", ["Male", "Female"])
stress_level = st.slider("Stress Level (1-6)", 1, 6, 3)
sleep_hours = st.number_input("Hours of Sleep per Night", min_value=1, max_value=12, value=7)
academic_pressure = st.slider("Academic Pressure (1-6)", 1, 6, 3)
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
social_support = st.slider("Social Support (1-6)", 1, 6, 4)
physical_activity = st.number_input("Hours of Physical Activity per Week", 0, 20, 5)
anxiety_level = st.slider("Anxiety Level (1-6)", 1, 6, 2)
diet_quality = st.slider("Diet Quality (1-6)", 1, 6, 4)
financial_stress = st.slider("Financial Stress (1-6)", 1, 6, 3)
extracurricular_activities = st.number_input("Hours of Extracurricular Activities per Week", 0, 20, 5)

# Submit Button
if st.button("Submit and Check"):
    try:
        # Predict risk and store in session_state
        st.session_state.risk = predict_risk(
            age, gender, stress_level, sleep_hours, academic_pressure,
            family_history, social_support, physical_activity,
            anxiety_level, diet_quality, financial_stress, extracurricular_activities
        )
        st.session_state.result = interpret_risk(st.session_state.risk)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display risk and feedback if prediction exists
if st.session_state.risk:
    st.success(f"Predicted Risk: {st.session_state.risk}")
    st.info(st.session_state.result)
    animated_risk_bar(st.session_state.risk)
    show_feedback(st.session_state.risk)

    # Optional Feature Importance
    if st.checkbox("Show Feature Importance"):
        st.subheader("Feature Importance")
        model, _, _ = load_model_and_encoders()
        features = [
            'age', 'gender', 'stress_level', 'sleep_hours', 'academic_pressure',
            'family_history', 'social_support', 'physical_activity',
            'anxiety_level', 'diet_quality', 'financial_stress', 'extracurricular_activities'
        ]
        importances = model.feature_importances_
        df_features = pd.DataFrame({'Feature': features, 'Importance': importances})
        df_features = df_features.sort_values(by='Importance', ascending=False)
        st.markdown(df_features.to_html(index=False, escape=False), unsafe_allow_html=True)

# Sample data
if st.checkbox("View Sample Training Data"):
    try:
        data = pd.read_csv('data/mental_health_data.csv')
        st.dataframe(data.head(10))
        st.subheader("Distribution of Risk Levels")
        st.bar_chart(data['mental_health_risk'].value_counts())
    except FileNotFoundError:
        st.warning("Sample data not found. Place it under 'data/mental_health_data.csv'.")

# Feedback
st.subheader("Feedback")
feedback = st.text_area("How was your experience?")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback! 💖")

st.info("⚠️ This tool provides an estimate. Please consult a professional for personalized advice.")