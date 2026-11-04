
import pandas as pd

# Load the diabetes dataset
data = pd.read_csv("diabetes_dataset.csv")

print("\n===== DATASET SIZE =====")
print(data.shape)

print("\n===== COLUMN NAMES =====")
print(data.columns.tolist())

print("\n===== DATA TYPES =====")
print(data.dtypes)

print("\n===== FIRST 5 ROWS =====")
print(data.head())

print("\n===== MISSING VALUES =====")
print(data.isnull().sum())

print("\n===== DIABETES TARGET DISTRIBUTION =====")
if "diabetes" in data.columns:
    print(data["diabetes"].value_counts())
else:
    print("Target column not found. Check the column names above.")
