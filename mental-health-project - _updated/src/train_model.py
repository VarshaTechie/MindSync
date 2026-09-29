import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# Load dataset
data = pd.read_csv('data/mental_health_data.csv')

# Encode categorical variables
le_gender = LabelEncoder()
le_family_history = LabelEncoder()
data['gender'] = le_gender.fit_transform(data['gender'])
data['family_history'] = le_family_history.fit_transform(data['family_history'])

# Features and target (updated with new columns)
X = data[['age', 'gender', 'stress_level', 'sleep_hours', 'academic_pressure', 'family_history', 'social_support', 'physical_activity', 'anxiety_level', 'diet_quality', 'financial_stress', 'extracurricular_activities']]
y = data['mental_health_risk']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Cross-validation scores
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Evaluate on test set
y_pred = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(model, 'models/mental_health_model.pkl')
joblib.dump(le_gender, 'models/gender_encoder.pkl')
joblib.dump(le_family_history, 'models/family_history_encoder.pkl')

print("Model and encoders saved.")
