# 🧠 HR Analytics: Predicting Employee Attrition

This project uses a machine learning model to predict employee attrition based on HR data. It includes:
- A classification model that outputs predictions into a CSV file
- An exploratory data analysis (EDA) chart showing attrition distribution

---

## 📂 Project Structure
 employee_attrition.csv # Input dataset
 attrition_predictions.csv # Output predictions
 run_model.py # Builds model and generates CSV
 plot_chart.py # Generates bar chart from dataset
 README.md # Project documentation

## How to Run

1. Make sure Python and required libraries are installed:
```
pip install pandas scikit-learn
```

2. Run the model script:
```
python run_attrition_model.py

```
✅ This will generate data/attrition_predictions.csv
3. Run the chart script:
```
python plot_chart.py
```
✅ This will show a bar chart of employee attrition distribution

🛠️ Tools & Technologies

Python
Pandas, scikit-learn
Matplotlib, Seaborn
Jupyter / VS Code
CSV Output (for use in Tableau / Power BI dashboards)

🧾 License

This project is for educational and demonstration purposes only. Dataset source: IBM HR Analytics (via Kaggle).
