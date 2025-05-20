# 📊 Client Subscribed to a Term Deposit

This project focuses on predicting whether a customer will subscribe to a term deposit based on the Bank Marketing dataset. It involves data preprocessing, visualization, and training multiple machine learning models.

---

## 🚀 Project Steps

### ✅ Step 1: Import Libraries
Essential libraries for data analysis, visualization, and machine learning were imported.

### ✅ Step 2: Load Dataset
The dataset `bankmarketing.csv` was loaded using `pandas`.

### ✅ Step 3: Data Cleaning
- Removed duplicates
- Handled missing values

### ✅ Step 4: Exploratory Data Analysis (EDA)
- Visualized target variable distribution
- Plotted categorical relationships (e.g., job vs. subscription)

### ✅ Step 5: Encode Categorical Variables
Used `LabelEncoder` to convert categorical columns to numerical format.

### ✅ Step 6: Feature Scaling
Normalized numerical features using `StandardScaler`.

### ✅ Step 7: Train-Test Split
Split the dataset into training and testing sets (80/20 split).

---

## 🧠 Models Trained

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

## 📈 Model Evaluation

Evaluated all models based on:
- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report

### 📊 Comparison Visualizations
- Bar plots comparing model performance
- ROC curves for each model
- Feature importance plots for tree-based models
- Decision Tree visualization using `plot_tree`

---

## 🖼️ Visual Examples

| Type                  | Description                          |
|-----------------------|--------------------------------------|
| 📉 Count Plots         | Target distribution, Job vs. Target  |
| 🔍 ROC Curve           | Model AUC comparison                 |
| 🌲 Decision Tree Plot  | Structure of tree splits             |
| 💡 Feature Importance | Top features influencing predictions |

---

## 📂 Dataset

- **File**: `bankmarketing.csv`
- **Target Column**: `y` (Yes/No - subscription to term deposit)

---

## 🛠️ Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `sklearn` (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, metrics, preprocessing, model_selection)

---

## ✅ Results Summary

Model: Logistic Regression
Accuracy: 0.9046867411364741
Precision: 0.6597938144329897
Recall: 0.39546858908341914
F1-Score: 0.49452672247263363
Classification Report:
               precision    recall  f1-score   support

           0       0.92      0.97      0.95      7265
           1       0.66      0.40      0.49       971

    accuracy                           0.90      8236
   macro avg       0.79      0.68      0.72      8236
weighted avg       0.89      0.90      0.89      8236


Model: Decision Tree
Accuracy: 0.8844099077221952
Precision: 0.5102260495156081
Recall: 0.4881565396498455
F1-Score: 0.4989473684210526
Classification Report:
               precision    recall  f1-score   support

           0       0.93      0.94      0.93      7265
           1       0.51      0.49      0.50       971

    accuracy                           0.88      8236
   macro avg       0.72      0.71      0.72      8236
weighted avg       0.88      0.88      0.88      8236


Model: Random Forest
Accuracy: 0.9108790675084992
Precision: 0.6612244897959184
Recall: 0.5005149330587023
F1-Score: 0.5697538100820633
Classification Report:
               precision    recall  f1-score   support

           0       0.94      0.97      0.95      7265
           1       0.66      0.50      0.57       971

    accuracy                           0.91      8236
   macro avg       0.80      0.73      0.76      8236
weighted avg       0.90      0.91      0.91      8236

