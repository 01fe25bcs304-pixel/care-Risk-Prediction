
# Install if needed:
# %pip install pandas numpy scikit-learn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.base import clone

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------
df = pd.read_csv("/kaggle/input/your-dataset/diabetic_data.csv")

TARGET = "readmitted"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

print("Dataset shape:", df.shape)
print("\nMissing values (top 15):")
print(df.isna().sum().sort_values(ascending=False).head(15))
print("\nDuplicate rows:", df.duplicated().sum())

# --------------------------------------------------
# 2. Prepare target and remove leakage columns
# --------------------------------------------------
# UCI dataset: '<30' = readmitted within 30 days.
# 'NO' and '>30' are negative for this target definition.
df = df.drop_duplicates().copy()
df = df.dropna(subset=[TARGET])

y = (df[TARGET].astype(str).str.strip() == "<30").astype(int)

# Exclude the target and identifiers that should not
# be used as predictive features.
drop_cols = [
    TARGET,
    "encounter_id",
    "patient_nbr"
]
X = df.drop(columns=drop_cols, errors="ignore")

# Drop columns that are entirely missing.
X = X.dropna(axis=1, how="all")

# Convert infinite numeric values to missing values.
X = X.replace([np.inf, -np.inf], np.nan)

if y.nunique() != 2:
    raise ValueError("The target must contain both positive and negative classes.")

if y.value_counts().min() < 3:
    raise ValueError("Not enough examples in one target class for stratified splitting.")

print("\nTarget distribution:")
print(y.value_counts(normalize=True).rename("proportion"))

# --------------------------------------------------
# 3. Split: 70% train, 10% validation, 20% test
# --------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Temp is 30% of total. Validation gets 1/3 of temp = 10%;
# test gets 2/3 of temp = 20%.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=2/3,
    random_state=42,
    stratify=y_temp
)

print("\nSplit sizes:")
print("Train:", len(X_train), f"({len(X_train)/len(df):.1%})")
print("Validation:", len(X_val), f"({len(X_val)/len(df):.1%})")
print("Test:", len(X_test), f"({len(X_test)/len(df):.1%})")

# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# --------------------------------------------------
# 5. Define three models
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=100,
        random_state=42
    )
}

# --------------------------------------------------
# 6. Train and select using validation results
# --------------------------------------------------
trained_models = {}
validation_results = []

for name, estimator in models.items():
    # HistGradientBoosting needs dense input.
    # For that model, use dense one-hot encoded features.
    if name == "HistGradientBoosting":
        dense_preprocessor = clone(preprocessor).set_params(
            categorical__encoder__sparse_output=False
        )
        pipeline = Pipeline([
            ("preprocessor", dense_preprocessor),
            ("model", estimator)
        ])
    else:
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", estimator)
        ])

    pipeline.fit(X_train, y_train)
    val_scores = pipeline.predict_proba(X_val)[:, 1]
    val_pred = (val_scores >= 0.5).astype(int)

    validation_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, val_pred),
        "Precision": precision_score(y_val, val_pred, zero_division=0),
        "Recall": recall_score(y_val, val_pred, zero_division=0),
        "F1-score": f1_score(y_val, val_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_val, val_scores),
        "Average Precision": average_precision_score(y_val, val_scores)
    })

    trained_models[name] = pipeline

validation_df = pd.DataFrame(validation_results)
print("\nVALIDATION RESULTS")
print(validation_df.round(4).to_string(index=False))

# Select using validation F1-score.
# For this healthcare task, also review recall and precision
# before finalizing the selection criterion.
best_name = validation_df.sort_values(
    ["F1-score", "Recall"], ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_name]
print("\nSelected model based on validation F1-score:", best_name)

# --------------------------------------------------
# 7. Final test evaluation (only after selection)
# --------------------------------------------------
test_scores = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_scores >= 0.5).astype(int)

print("\nFINAL TEST RESULTS")
print("Model:", best_name)
print(f"Accuracy:          {accuracy_score(y_test, test_pred):.4f}")
print(f"Precision:         {precision_score(y_test, test_pred, zero_division=0):.4f}")
print(f"Recall:            {recall_score(y_test, test_pred, zero_division=0):.4f}")
print(f"F1-score:          {f1_score(y_test, test_pred, zero_division=0):.4f}")
print(f"ROC-AUC:           {roc_auc_score(y_test, test_scores):.4f}")
print(f"Average Precision: {average_precision_score(y_test, test_scores):.4f}")

print("\nClassification report:")
print(classification_report(
    y_test, test_pred,
    target_names=["Not readmitted within 30 days", "Readmitted within 30 days"],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test, test_pred))

RocCurveDisplay.from_predictions(y_test, test_scores)
plt.title(f"ROC Curve - {best_name}")
plt.tight_layout()
plt.show()
