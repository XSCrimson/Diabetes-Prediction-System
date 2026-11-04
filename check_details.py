
import pandas as pd

data = pd.read_csv("diabetes_dataset.csv")

print("\n===== CATEGORICAL VALUES =====")

for column in ["gender", "smoking_history"]:
    print(f"\n{column}:")
    print(data[column].unique())

print("\n===== NUMERICAL RANGES =====")

columns = [
    "age",
    "bmi",
    "hbA1c_level",
    "blood_glucose_level"
]

print(data[columns].describe())

print("\n===== BINARY FEATURES =====")

for column in ["hypertension", "heart_disease"]:
    print(f"\n{column}:")
    print(data[column].value_counts())

print("\n===== DUPLICATE RECORDS =====")
print(data.duplicated().sum())
