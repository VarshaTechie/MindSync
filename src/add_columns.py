import pandas as pd
import numpy as np

# Load the existing data
data = pd.read_csv('data/mental_health_data.csv')

# Add new columns with random values
np.random.seed(42)  # For reproducibility
data['diet_quality'] = np.random.randint(1, 7, size=len(data))  # 1-6 scale
data['financial_stress'] = np.random.randint(1, 7, size=len(data))  # 1-6 scale
data['extracurricular_activities'] = np.random.randint(0, 21, size=len(data))  # 0-20 hours

# Save the updated data
data.to_csv('data/mental_health_data.csv', index=False)
print("Added new columns: diet_quality, financial_stress, extracurricular_activities")
