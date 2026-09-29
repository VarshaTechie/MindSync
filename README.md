# Student Mental Health Survey

A Streamlit-based web application that predicts mental health risk levels for students based on various factors.

## Features

- Predict mental health risk (Low, Medium, High) using a trained Random Forest model.
- Interactive UI with sliders and inputs for user data.
- Visualizations: Feature importance and risk level distribution.
- Feedback section for user input.
- Disclaimer for responsible use.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd mental-health-project
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the training script to train the model:
   ```
   python src/train_model.py
   ```

4. Run the Streamlit app:
   ```
   streamlit run src/app.py
   ```

## Data

The dataset includes features like age, gender, stress level, sleep hours, academic pressure, family history, social support, physical activity, anxiety level, diet quality, financial stress, and extracurricular activities.

## Model

- Uses Random Forest Classifier with hyperparameter tuning via GridSearchCV.
- Evaluated with accuracy, precision, recall, F1-score, and confusion matrix.

## Deployment

Deploy on Streamlit Cloud or similar platforms by uploading the code and ensuring models are in the `models/` directory.

## Disclaimer

This tool is for educational purposes only and not a substitute for professional medical advice.
