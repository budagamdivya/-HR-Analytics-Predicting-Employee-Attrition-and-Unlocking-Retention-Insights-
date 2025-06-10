# HR Analytics: Predicting Employee Attrition

## 📝 Project Overview
This project analyzes employee data to predict attrition using machine learning and visualize insights through an interactive dashboard. Built using Python (pandas, scikit-learn), SHAP for model interpretability, and Power BI for visualization.


## 🔧 Tools & Technologies
- Python: Data wrangling, machine learning (Logistic Regression, Random Forest), SHAP for feature importance
- Power BI: Visualization dashboard
- Jupyter/VSCode: Development environment
- Kaggle Dataset: IBM HR Analytics

## 📊 Key Features
- Predict employee attrition using supervised ML models
- Visualize trends in attrition by department, job role, tenure
- Identify top factors contributing to attrition using SHAP
- Dashboard for HR managers to make data-driven decisions

## 📂 Dataset
**Source**: [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## 🚀 How to Run
1. Clone this repository.
2. Download the dataset and place it in `Employee Attrition.csv`.
3. Install dependencies:
   ```bash
   pip install pandas scikit-learn shap matplotlib seaborn
   ```
4. Run the project script:
   ```bash
   python hr_attrition_project.py
   ```
5. Open Power BI / Tableau and load `attrition predictions.csv` to create the dashboard.

## 📊 Dashboard Ideas
- Attrition % by Department, Gender, Job Role
- Risk scoring chart using model predictions
- Heatmaps by Age vs Tenure
- Filters for engagement levels and overtime

## 🧠 Business Insights (Example)
- High overtime and low satisfaction = 3x more likely to churn
- Sales and HR roles show higher attrition under 3 years of service
- Top predictors: MonthlyIncome, JobSatisfaction, OverTime, EnvironmentSatisfaction

## 🧾 License
This project is for educational and portfolio purposes. Data sourced from IBM via Kaggle.

---

# hr_attrition_project.py

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import shap

# Step 2: Load Dataset
df = pd.read_csv('Employee Attrition.csv')

# Step 3: Data Cleaning
# Convert target variable to binary
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Drop columns that are not useful
df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)

# Handle categorical variables
df = pd.get_dummies(df, drop_first=True)

# Step 4: EDA (Example Plot)
sns.countplot(x='Attrition', data=df)
plt.title('Attrition Count')
plt.show()

# Step 5: Feature and Target Split
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train Models
log_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Step 8: Evaluation
y_pred_log = log_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_log))

print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

# Step 9: Feature Importance (SHAP for Random Forest)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values[1], X_train)

# Step 10: Save Processed Data for Dashboard
X_test['Attrition_Prediction'] = y_pred_rf
X_test.to_csv('data/attrition_predictions.csv', index=False)

# Step 11: Dashboard
# Open Power BI / Tableau, import `attrition_predictions.csv`
# Create visuals like:
# - Attrition by Department
# - Attrition by Age or Tenure
# - Feature importance (import image from SHAP summary)
# - Filter by Job Role, Gender, Monthly Income

# Note: Ensure to save dashboard file in dashboards/ folder
# e.g., dashboards/HR_attrition_dashboard.pbix
