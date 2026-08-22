# 🌾 Crop Field AI

## AI-Powered Crop Recommendation and Yield Prediction System

Crop Field AI is a machine-learning-based agricultural decision-support application that provides:

- 🌱 Crop recommendation based on soil and environmental conditions
- 📊 Crop yield prediction based on historical agricultural data
- 🔍 Top-3 crop predictions with probability scores
- 📈 Estimated total crop output based on predicted yield and cultivated area
- 🤖 Random Forest machine-learning models
- 🌐 Interactive Streamlit web application

---

## 🚀 Live Application

The application is deployed using Streamlit Community Cloud.

**Live Demo:**  
https://crop-recommendation-and-yield-prediction-system-ymxndqgfkmw2y.streamlit.app/

---

## ✨ Features

### 1. Crop Recommendation

The system recommends the most suitable crop using:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

The application provides:

- Best crop recommendation
- Top-3 crop predictions
- Prediction probabilities
- Model uncertainty information

### 2. Yield Prediction

The system estimates crop yield using:

- Crop year
- Crop
- Season
- State
- Annual rainfall
- Fertilizer quantity
- Pesticide quantity
- Cultivated area

The application provides:

- Predicted yield per unit area
- Estimated total output
- Input summary
- Model information

---

## 🧠 Machine Learning Models

### Crop Recommendation

**Algorithm:** Random Forest Classifier

The classification model predicts the most suitable crop from the provided soil and environmental conditions.

### Yield Prediction

**Algorithm:** Random Forest Regressor

The regression model predicts agricultural yield using historical crop-production data.

---

## 📊 Model Evaluation

### Crop Recommendation Model

The model was evaluated using a held-out test dataset containing **440 samples**.

| Metric | Score |
|---|---:|
| Accuracy | 99.32% |
| Weighted Precision | 99.35% |
| Weighted Recall | 99.32% |
| Weighted F1 Score | 99.32% |

The confusion matrix showed only a small number of misclassifications across the crop classes.

### Yield Prediction Model

| Metric | Score |
|---|---:|
| MAE | 9.6624 |
| RMSE | 129.6144 |
| R² | 0.9790 |

The yield model explains approximately **97.9% of the variance** in the test data.

---

## 🔄 System Workflow

```text
User Input
    ↓
Input Validation
    ↓
Data Preprocessing
    ↓
Trained ML Model
    ↓
Prediction
    ↓
Result + Explanation
