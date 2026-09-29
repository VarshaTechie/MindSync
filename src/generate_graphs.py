import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Load dataset
data = pd.read_csv('data/mental_health_data.csv')

# Encode categorical variables (same as in train_model.py)
le_gender = LabelEncoder()
le_family_history = LabelEncoder()
data['gender'] = le_gender.fit_transform(data['gender'])
data['family_history'] = le_family_history.fit_transform(data['family_history'])

# Features and target
X = data[['age', 'gender', 'stress_level', 'sleep_hours', 'academic_pressure', 'family_history', 'social_support', 'physical_activity', 'anxiety_level', 'diet_quality', 'financial_stress', 'extracurricular_activities']]
y = data['mental_health_risk']

# Split data (same as in train_model.py for consistency)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model
model = joblib.load('models/mental_health_model.pkl')

# Predict on test set for confusion matrix
y_pred = model.predict(X_test)

# 1. Feature Importance Bar Chart
features = [
    'age', 'gender', 'stress_level', 'sleep_hours', 'academic_pressure',
    'family_history', 'social_support', 'physical_activity',
    'anxiety_level', 'diet_quality', 'financial_stress', 'extracurricular_activities'
]
importances = model.feature_importances_
df_features = pd.DataFrame({'Feature': features, 'Importance': importances})
df_features = df_features.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=df_features, hue='Feature', palette='viridis', legend=False)
plt.title('Feature Importance in Mental Health Risk Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# 2. Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix for Mental Health Risk Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# 3. Risk Level Distribution Pie Chart (based on predictions on the full dataset)
full_predictions = model.predict(X)
risk_counts = pd.Series(full_predictions).value_counts()

plt.figure(figsize=(8, 8))
plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=140, colors=['#81C784', '#FFF176', '#E57373'])
plt.title('Predicted Risk Level Distribution (Full Dataset)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# 4. Line Chart: Accuracy vs. Number of Estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

n_estimators_range = [10, 20, 50, 100, 150, 200]
train_accuracies = []
val_accuracies = []

for n_est in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
    rf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    val_acc = accuracy_score(y_test, rf.predict(X_test))
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(n_estimators_range, val_accuracies, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy vs. Number of Estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 5. Overfitting/Underfitting: Training and Validation Accuracy vs. Max Depth
max_depth_range = [1, 2, 5, 10, 15, None]
train_accuracies_depth = []
val_accuracies_depth = []

for depth in max_depth_range:
    rf = RandomForestClassifier(max_depth=depth, random_state=42)
    rf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    val_acc = accuracy_score(y_test, rf.predict(X_test))
    train_accuracies_depth.append(train_acc)
    val_accuracies_depth.append(val_acc)

plt.figure(figsize=(10, 6))
plt.plot([str(d) if d is not None else 'None' for d in max_depth_range], train_accuracies_depth, label='Training Accuracy', marker='o')
plt.plot([str(d) if d is not None else 'None' for d in max_depth_range], val_accuracies_depth, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy vs. Max Depth (Overfitting Check)')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 6. Validation Error (1 - Accuracy) Line Chart
val_errors = [1 - acc for acc in val_accuracies]
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, val_errors, label='Validation Error', marker='o', color='red')
plt.title('Validation Error vs. Number of Estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Error (1 - Accuracy)')
plt.legend()
plt.grid(True)
plt.show()
