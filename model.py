import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Extended_Employee_Performance_and_Productivity_Data.csv")

# Select features and target
features = ['Years_At_Company', 'Monthly_Salary', 'Overtime_Hours', 'Promotions', 'Employee_Satisfaction_Score']
target = 'Performance_Score'
X = df[features]
y = df[target]

# Preprocess and split the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the scaler and model
joblib.dump(scaler, "Scaler.pkl")
joblib.dump(model, "model.pkl")
print("Scaler.pkl and model.pkl generated successfully.")
