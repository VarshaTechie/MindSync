# TODO List for Implementing Mental Health Questionnaire

- [x] Define questionnaire data structure with categories, questions, and mappings in app.py
- [x] Replace sliders with st.tabs for each category (Stress Level, Academic Pressure, Social Support, Anxiety Level, Diet Quality, Financial Stress)
- [x] In each tab, add radio buttons for questions (1-3 scale)
- [x] Collect responses and store in session state
- [x] On submit, calculate sum per category, map to 1-6 score
- [x] Use the mapped scores for predict_risk instead of direct sliders
- [x] Update display to show results based on questionnaire
- [x] Test the app to ensure functionality
