# Telecom Churn Predictor

## Overview

This project focuses on predicting customer churn using machine learning models. By analyzing a dataset of customer behaviors and service usage, the project identifies key patterns and provides actionable insights to help businesses retain at-risk customers. The primary objective is to minimize churn by proactively targeting customers likely to leave.

---

## Objectives

- **Churn Prediction**: Predict customers likely to churn using advanced machine learning models.
- **Customer Insights**: Identify key characteristics and behaviors that correlate with churn.
- **Business Impact**: Enable proactive retention strategies by prioritizing high-risk customers.
- 
![image](https://github.com/user-attachments/assets/cb5b31ca-31ee-488d-a4dd-1e4c5729a987)

---

## Dataset

The dataset consists of 5,000 customer records, including:
- **Categorical Features**: `state`, `international_plan`, `voice_mail_plan`.
- **Numerical Features**: Usage metrics like `total_day_minutes`, `total_night_calls`, and more.
- **Target Variable**: `Churn` (binary classification: 1 for churn, 0 for no churn).

The dataset is clean, with no missing values and features already preprocessed.

---

## Methodology

### Data Preprocessing
1. Dropped redundant features (e.g., `phone_number`).
2. Encoded categorical variables using `LabelEncoder`.
3. Balanced the target variable using class weights and stratified sampling.

### Exploratory Data Analysis (EDA)
- Visualized customer distribution across features.
- Analyzed correlations to identify relationships between variables and churn.

### Machine Learning Models
The following models were trained and evaluated:
1. **Logistic Regression**
   - Scaled features using `StandardScaler`.
   - Class weights handled imbalance in `Churn`.
2. **Random Forest**
   - Leveraged for feature importance and robust performance.
   - No scaling required due to tree-based nature.
3. **XGBoost**
   - Tuned `scale_pos_weight` to address class imbalance.
   - Achieved the highest recall and best overall performance.

---

## Key Results

![image](https://github.com/user-attachments/assets/14730bd4-4955-472f-acdf-db4a172147fc)

- **Best Model**: XGBoost achieved a recall of 0.84, outperforming other models.
- **Feature Importance**:
  - Key drivers of churn include `international_plan`, `number_customer_service_calls`, and `voice_mail_plan`.
- **Churn Strategy**: Ranked top 500 customers with the highest churn probability for proactive engagement.

![image](https://github.com/user-attachments/assets/ee4b1900-645a-490b-b8ab-0a901aa8a96a)

---

## Tools and Technologies

- **Programming**: Python (Pandas, NumPy, Scikit-learn, XGBoost).
- **Visualization**: Matplotlib, Seaborn.
- **Model Evaluation**: Confusion matrix, ROC curve, classification report.
- **Cross-Validation**: Recall as the primary metric for imbalanced classification.

---

## Visualizations

- **Correlation Matrix**: Explored relationships between features.
- **Feature Importance**: Highlighted key predictors of churn.
- **Customer Segmentation**: Visualized churn probabilities by feature subsets.

---

## Future Work

- Hyperparameter tuning for optimal model performance.
- Deploy the model for real-time churn prediction.
- Integrate external factors (e.g., customer demographics) for enhanced insights.

---

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/churn-prediction.git
   cd churn-prediction
