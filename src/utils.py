import pandas as pd
import joblib

def load_model_and_encoders():
    """Load the trained model and encoders."""
    model = joblib.load('models/mental_health_model.pkl')
    le_gender = joblib.load('models/gender_encoder.pkl')
    le_family_history = joblib.load('models/family_history_encoder.pkl')
    return model, le_gender, le_family_history

def predict_risk(age, gender, stress_level, sleep_hours, academic_pressure, family_history, social_support, physical_activity, anxiety_level, diet_quality, financial_stress, extracurricular_activities):
    """Predict mental health risk based on input features."""
    model, le_gender, le_family_history = load_model_and_encoders()
    gender_encoded = le_gender.transform([gender])[0]
    family_encoded = le_family_history.transform([family_history])[0]
    input_data = pd.DataFrame([[age, gender_encoded, stress_level, sleep_hours, academic_pressure, family_encoded, social_support, physical_activity, anxiety_level, diet_quality, financial_stress, extracurricular_activities]],
                              columns=['age', 'gender', 'stress_level', 'sleep_hours', 'academic_pressure', 'family_history', 'social_support', 'physical_activity', 'anxiety_level', 'diet_quality', 'financial_stress', 'extracurricular_activities'])
    prediction = model.predict(input_data)[0]
    return prediction
