import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score

dataset = pd.read_csv(r"data/diabetes_dataset.csv")
train_data = pd.read_csv(r"data/train.csv")
test_data = pd.read_csv(r"data/test.csv")

full_data = pd.concat([train_data, dataset], join="inner", ignore_index=True)

labels = full_data["diagnosed_diabetes"]
diabetes_data = full_data.drop("diagnosed_diabetes", axis=1)

# Feature engineering (add relevant features based on your data)
# Example: if you have these columns
numeric_cols = diabetes_data.select_dtypes(include=["int64", "float64"]).columns
if len(numeric_cols) >= 2:
    # Create interaction features for top numeric columns
    for i, col1 in enumerate(numeric_cols[:3]):
        for col2 in numeric_cols[i + 1 : 4]:
            diabetes_data[f"{col1}_{col2}_interaction"] = (
                diabetes_data[col1] * diabetes_data[col2]
            )

# Split first, then encode
x_train, x_test, y_train, y_test = train_test_split(
    diabetes_data,
    labels,
    shuffle=True,
    test_size=0.1,
    stratify=labels,
    random_state=42,
)

# Encode categorical variables
cat_cols = x_train.select_dtypes(include=["object"]).columns
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    x_train[col] = le.fit_transform(x_train[col])
    x_test[col] = le.transform(x_test[col])
    label_encoders[col] = le

# Quick hyperparameter tuning
param_grid = {
    "max_depth": [5, 6, 7],
    "learning_rate": [0.05, 0.07, 0.1],
    "min_child_weight": [1, 3, 5],
    "scale_pos_weight": [0.7, 0.8, 0.9, 1.0],
}

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring="f1_macro",
    verbose=1,
    n_jobs=-1,
)

print("Running grid search...")
grid_search.fit(x_train, y_train)
print("Best parameters:", grid_search.best_params_)

# Train final model with best params
model = grid_search.best_estimator_

# Get probability predictions for threshold tuning
y_pred_proba = model.predict_proba(x_test)[:, 1]

# Find optimal threshold
thresholds = np.arange(0.35, 0.65, 0.01)
f1_scores = []

for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_threshold, average="macro"))

optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"\nOptimal threshold: {optimal_threshold:.3f}")

# Final predictions with optimal threshold
predictions = (y_pred_proba >= optimal_threshold).astype(int)

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy:.6f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Feature importance
feature_importance = pd.DataFrame(
    {"feature": x_train.columns, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
