
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load dataset
data = pd.read_csv("diabetes_dataset.csv")

# 2. Remove duplicate records
data = data.drop_duplicates()

# 3. Select input features
features = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "hbA1c_level",
    "blood_glucose_level"
]

X = data[features]
y = data["diabetes"]

# 4. Identify categorical and numerical columns
categorical_features = [
    "gender",
    "smoking_history"
]

numerical_features = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "hbA1c_level",
    "blood_glucose_level"
]

# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# 6. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=25,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1
)

# 7. Create pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)

# 8. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. Train model
print("Training diabetes prediction model...")

pipeline.fit(X_train, y_train)

# 10. Make predictions
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# 11. Evaluate original model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Diabetes", "Diabetes"],
        zero_division=0
    )
)

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

# 12. Compare different prediction thresholds
print("\n===== THRESHOLD COMPARISON =====")

thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25]

for threshold in thresholds:

    predictions = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

# 13. Save trained model
joblib.dump(pipeline, "diabetes_model.pkl")

print("\nModel successfully saved as diabetes_model.pkl")
