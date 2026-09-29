import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.model_selection import train_test_split
import sys
import os

# Fix path for utils
sys.path.append(os.path.dirname(__file__))
from utils import load_model_and_encoders

# Load data
df = pd.read_csv('data/mental_health_data.csv')

# Preprocess data (similar to train_model.py)
le_gender = LabelEncoder()
le_family_history = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])
df['family_history'] = le_family_history.fit_transform(df['family_history'])
df = df.drop(['id'], axis=1, errors='ignore')  # Drop id if present

X = df.drop('mental_health_risk', axis=1)
y = df['mental_health_risk']

# Load model and encoders
model, family_encoder, gender_encoder = load_model_and_encoders()

# Predictions on full dataset
predictions = model.predict(X)
probabilities = model.predict_proba(X)

# 1. Confusion Matrix
cm = confusion_matrix(y, predictions, labels=['Low', 'Medium', 'High'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'], cmap='Blues')
plt.title('Confusion Matrix (Actual vs Predicted)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 2. ROC Curve (One-vs-Rest for multiclass)
y_bin = label_binarize(y, classes=['Low', 'Medium', 'High'])
n_classes = y_bin.shape[1]

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], probabilities[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{["Low", "Medium", "High"][i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (One-vs-Rest)')
plt.legend(loc="lower right")
plt.show()

# 3. Precision-Recall Curve (One-vs-Rest for multiclass)
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], probabilities[:, i])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{["Low", "Medium", "High"][i]} (AUC = {pr_auc:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (One-vs-Rest)')
plt.legend(loc="lower left")
plt.show()
