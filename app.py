# 👥 Employee Performance & Productivity Analysis

An end-to-end machine learning project that analyzes 100,000 employee records to identify key drivers of performance and build a predictive model for employee Performance Scores — deployed via an interactive Streamlit web application.

---

## 📌 Problem Statement

HR departments struggle to identify early signals of high or low employee performance across large workforces. Manual reviews are slow, inconsistent, and miss cross-departmental patterns. This project uses exploratory data analysis and machine learning to:
- Identify which factors most strongly influence Performance Scores
- Build a model that predicts an employee's Performance Score (1–5) from their profile
- Deliver predictions through a live Streamlit app, enabling HR teams to run instant assessments

---

## 📊 Dashboard / App Preview

![Streamlit App Screenshot](Assets/Screenshot%202025-01-21%20093046.png)
![Dashboard View](Assets/Screenshot%202025-01-21%20093103.png)

---

## 🗃️ Dataset

| Property | Detail |
|---|---|
| Source | Extended Employee Performance and Productivity Dataset |
| Rows | 100,000 |
| Columns | 20 |
| Target Variable | `Performance_Score` (ordinal: 1 = lowest, 5 = highest) |

**Key features:** `Department`, `Gender`, `Age`, `Job_Title`, `Years_At_Company`, `Education_Level`, `Monthly_Salary`, `Projects_Handled`, `Overtime_Hours`, `Sick_Days`, `Remote_Work_Frequency`, `Team_Size`, `Training_Hours`, `Promotions`, `Employee_Satisfaction_Score`, `Resigned`

---

## 🔍 Key EDA Insights

- **Monthly Salary is the strongest predictor** of Performance Score, with a correlation of **r = 0.51** — far above all other features
- **Marketing** department has the highest average salary; **Customer Support** has the lowest
- **IT department** employees log the highest average overtime hours (~14.7 hrs/week)
- Employees who have **not resigned** tend to cluster at Performance Scores 3–5
- Gender distribution is nearly balanced: Male (48%), Female (48%), Other (4%)
- Performance Score distribution is approximately uniform across all 5 levels (~20% each)

---

## 🤖 ML Modelling

### Features Used
| Feature | Rationale |
|---|---|
| `Monthly_Salary` | Strongest correlation with Performance Score (r = 0.51) |
| `Years_At_Company` | Seniority proxy — experience level |
| `Overtime_Hours` | Work intensity signal |
| `Promotions` | Track record of recognised performance |
| `Employee_Satisfaction_Score` | Engagement and motivation indicator |

### Model Comparison

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 30.6% | Baseline; struggles with non-linear boundaries |
| KNN (GridSearchCV) | **49.8%** | Best params: `n_neighbors=3`, `weights='distance'` |
| SVM (GridSearchCV) | In progress | Tested kernels: `linear`, `rbf`, `poly` |

> **Why is accuracy moderate?** The dataset is synthetically generated with near-uniform distribution across 5 performance classes and weak inter-feature correlations. In a real-world HR dataset with stronger signal, these models would perform significantly better. The key analytical finding — that salary is the dominant driver — remains valid and actionable.

### Data Preprocessing
- Dropped low-signal columns: `Employee_ID`, `Hire_Date`, `Work_Hours_Per_Week`, `Sick_Days`, `Remote_Work_Frequency`, `Team_Size`, `Training_Hours`, `Resigned`
- No missing values or duplicate records found
- Applied `StandardScaler` to all numeric features before model training
- 80/20 train-test split

---

## 🚀 Streamlit App

The app allows HR teams to input an employee's profile and instantly receive a predicted Performance Score (1–5).

**Inputs:** Years at company, Monthly salary, Overtime hours, Number of promotions, Employee satisfaction score

**Output:** Predicted Performance Score category

### Run Locally

```bash
git clone https://github.com/RamcharanSinghRamavath/Employee-Performance-Analysis.git
cd Employee-Performance-Analysis
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| Pandas, NumPy | Data manipulation and EDA |
| Matplotlib, Seaborn | Visualisation |
| Scikit-learn | ML models, preprocessing, GridSearchCV |
| Streamlit | Interactive web app deployment |
| Joblib | Model serialisation (scaler.pkl, model.pkl) |

---

## 📁 Repository Structure

```
Employee-Performance-Analysis/
│
├── Employee Performance Analysis.ipynb   # Full EDA + modelling notebook
├── app.py                                 # Streamlit prediction app
├── model.py                               # Model training script
├── model.pkl                              # Saved SVM model
├── Scaler.pkl                             # Saved StandardScaler
├── Employee_Performance_and_Productivity_Dataset.csv
├── requirements.txt
├── Assets/
│   └── Screenshots
└── README.md
```

---

## 💡 Business Implications

- **Salary alignment** is the single biggest lever for performance — HR should audit pay equity across departments before investing in training programmes
- Employees with 3+ promotions show consistently higher performance clusters — early recognition matters
- Overtime hours show minimal correlation with performance, suggesting that longer hours ≠ higher output

---

## 🔮 Future Enhancements

- [ ] Deploy app on Streamlit Cloud with public URL
- [ ] Add Random Forest and XGBoost for performance comparison
- [ ] Incorporate feature engineering (e.g., salary-to-tenure ratio, promotion rate)
- [ ] Build attrition prediction model using the `Resigned` column
- [ ] Add SHAP values for model explainability

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
