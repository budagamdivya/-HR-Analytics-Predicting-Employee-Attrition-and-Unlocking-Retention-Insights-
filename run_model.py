import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("data/employee_attrition.csv")

# Preprocess
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

# Split data
X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Save CSV
os.makedirs("data", exist_ok=True)
X_test_copy = X_test.copy()
X_test_copy['Attrition_Prediction'] = y_pred
output_path = "data/attrition_predictions.csv"
X_test_copy.to_csv(output_path, index=False)

# Confirm save
if os.path.exists(output_path):
    print(f"\n✅ CSV file saved successfully at: {output_path}")
else:
    print(f"\n❌ Failed to save CSV file.")