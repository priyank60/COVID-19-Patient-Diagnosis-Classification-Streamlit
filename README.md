readme_content = """
# 🦠 COVID-19 Patient Symptoms & Diagnosis Prediction

## 📌 Project Overview

This project focuses on predicting COVID-19 diagnosis (Positive / Negative) using patient symptoms, demographic information, and clinical indicators.  
The primary objective is to maximize recall to ensure no COVID-positive patient is missed, which is critical in healthcare decision-making.

A complete end-to-end machine learning pipeline was implemented, including data preprocessing, feature engineering, model training, evaluation, and deployment using Streamlit.

---

## 🎯 Problem Statement

Early detection of COVID-19 is essential to reduce transmission and ensure timely treatment.  
False negatives can be dangerous in medical diagnosis.

### Goals of this system:
- Predict COVID-19 test results using patient data
- Identify important features influencing diagnosis
- Compare multiple classification models
- Deploy the best-performing model using Streamlit

---

## 📂 Project Folder Structure

COVID-19-Patient-Diagnosis/
│
├── DataSets/
│   └── covid19_patient_symptoms_diagnosis.csv
│
├── Images/
│   └── Dashboard_image.png
│
├── Notebook/
│   └── COVID_19_Patient_Symptoms_And_Diagnosis.ipynb
│
├── Pickel_files/
│   ├── columns.pkl
│   ├── logistic_model.pkl
│   └── scaler.pkl
│
├── Streamlit_app/
│   └── covid19_app.py
│
├── requirements.txt
└── README.md



---

## 📊 Dataset Description

- **File:** covid19_patient_symptoms_diagnosis.csv  
- **Records:** 5,000  
- **Features:** 18  

### Key Columns:
- **Demographics:** age, gender  
- **Symptoms:** fever, dry_cough, sore_throat, fatigue, headache,  
  shortness_of_breath, loss_of_smell, loss_of_taste, chest_pain  
- **Clinical data:** oxygen_level, body_temperature  
- **Health & exposure:** low_oxygen, fever_temp, travel_history, contact_with_patient  
- **Target:** covid_result (1 = Positive, 0 = Negative)

---

## 🧹 Data Preprocessing & Feature Engineering

- Missing values in `comorbidity` were replaced with **"Unknown"**
- No duplicate records found
- Feature engineering:
  - oxygen_level → low_oxygen (1 if < 94)
  - body_temperature → fever_temp (1 if > 37.5°C)
- Encoded categorical variables (gender, comorbidity)
- Applied scaling using `StandardScaler`
- Saved trained preprocessing objects as pickle files

---

## 🤖 Models Trained

- Logistic Regression 
- Decision Tree 
- Random Forest 
- XGBoost 
- Support Vector Machine 
- K-Nearest Neighbors 
- Naive Bayes 

(All models evaluated in original and tuned versions)

---

## 📊 Complete Classification Model Comparison

| Accuracy | Precision | Recall | F1-Score | Log-Loss | ROC-AUC | Model |
|--------|-----------|--------|----------|----------|---------|-------|
| 0.95 | 0.91 | 1.00 | 0.96 | 0.1269 | 0.99 | Logistic Regression |
| 0.95 | 0.92 | 1.00 | 0.96 | 0.2575 | 0.98 | Tuned Random Forest |
| 0.95 | 0.91 | 1.00 | 0.96 | 0.1292 | 0.98 | Tuned SVM Model |
| 0.95 | 0.91 | 1.00 | 0.96 | 0.1361 | 0.98 | SVM Model |
| 0.95 | 0.91 | 1.00 | 0.96 | 0.2066 | 0.98 | Tuned XGBoost Classifier |
| 0.93 | 0.88 | 0.99 | 0.94 | 0.2769 | 0.97 | Naive Bayes Model |
| 0.93 | 0.88 | 0.99 | 0.94 | 0.2769 | 0.97 | Tuned Naive Bayes Model |
| 0.94 | 0.93 | 0.96 | 0.95 | 0.3641 | 0.98 | Tuned Decision Tree |
| 0.94 | 0.92 | 0.96 | 0.94 | 0.2020 | 0.98 | Random Forest |
| 0.93 | 0.92 | 0.94 | 0.93 | 0.1802 | 0.98 | XGBoost Classifier |
| 0.91 | 0.92 | 0.91 | 0.92 | 3.1358 | 0.91 | Decision Tree |
| 0.90 | 0.90 | 0.89 | 0.90 | 0.3066 | 0.96 | Tuned KNN Model |
| 0.86 | 0.86 | 0.86 | 0.86 | 1.1979 | 0.91 | KNN Model |

---

## 🏆 Best Model Performance

### Logistic Regression (Final Model)

| Metric | Score |
|------|------|
| Accuracy | 0.95 |
| Precision | 0.91 |
| Recall | 1.00 |
| F1-Score | 0.96 |
| Log-Loss | 0.1269 |
| ROC-AUC | 0.99 |

### 📌 Why Logistic Regression?
- Perfect recall (no missed COVID-positive cases)
- Stable probability estimates
- High interpretability
- Ideal for medical diagnosis use cases

---

## 📸 Streamlit Dashboard Preview

![Streamlit Dashboard](Images/Dashboard_image.png)

---

## 🌐 Streamlit Web Application

The trained model is deployed using Streamlit for interactive predictions.

### ▶️ Run the App Locally

```bash
streamlit run Streamlit_app/covid19_app.py
```
---

## 👤 Author

**Name:** Priyank Shrivastava
**Email:** priyankshrivastava5678@gmail.com

"""

[def]: Images/Dashboard_image.png