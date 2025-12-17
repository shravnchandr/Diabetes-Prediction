import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
dataset = pd.read_csv(r"data/diabetes_dataset.csv")
train_data = pd.read_csv(r"data/train.csv")
test_data = pd.read_csv(r"data/test.csv")

extra_features = [
    "glucose_postprandial",
    "insulin_level",
    "glucose_fasting",
    "diabetes_risk_score",
    "diabetes_stage",
    "hba1c",
]

# Get common features
common_features = [
    col
    for col in dataset.columns
    if col in train_data.columns and col != "diagnosed_diabetes"
]

print("=" * 60)
print("STEP 1: Training Imputation Models")
print("=" * 60)

# Train imputation models
imputation_models = {}

for feature in extra_features:
    if feature not in dataset.columns:
        continue

    mask = dataset[feature].notna()
    if mask.sum() < 10:
        print(f"Skipping {feature} - insufficient data")
        continue

    X = dataset.loc[mask, common_features].copy()
    y = dataset.loc[mask, feature]

    # Encode categorical
    X_encoded = X.copy()
    cat_cols = X_encoded.select_dtypes(include=["object"]).columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le

    # Train model
    is_categorical = y.dtype == "object" or y.nunique() < 20

    if is_categorical:
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y.astype(str))
        model = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        )
        model.fit(X_encoded, y_encoded)
        encoders["target"] = le_target
    else:
        model = RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        )
        model.fit(X_encoded, y)

    imputation_models[feature] = {
        "model": model,
        "encoders": encoders,
        "is_categorical": is_categorical,
    }
    print(f"✓ Trained model for {feature}")

print("\n" + "=" * 60)
print("STEP 2: Imputing Missing Features")
print("=" * 60)


# Function to impute
def impute_data(data, models, common_features):
    result = data.copy()

    for feature, model_dict in models.items():
        model = model_dict["model"]
        encoders = model_dict["encoders"]
        is_categorical = model_dict["is_categorical"]

        X = data[common_features].copy()

        # Encode categoricals
        for col, encoder in encoders.items():
            if col == "target" or col not in X.columns:
                continue
            X[col] = (
                X[col]
                .astype(str)
                .apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
            )
            X[col] = encoder.transform(X[col])

        # Predict
        predictions = model.predict(X)

        # Decode if categorical
        if is_categorical and "target" in encoders:
            predictions = encoders["target"].inverse_transform(predictions)

        result[feature] = predictions
        print(f"✓ Imputed {feature}")

    return result


train_data_imputed = impute_data(train_data, imputation_models, common_features)
test_data_imputed = impute_data(test_data, imputation_models, common_features)

# Combine all data
full_data = pd.concat([train_data_imputed, dataset], ignore_index=True)

print(f"\n✓ Full data shape: {full_data.shape}")
print(f"✓ Total features: {len(full_data.columns) - 1}")

# Continue with training
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import (
    StackingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score

# ... (your existing imputation code) ...

# After getting full_data with imputed features
labels = full_data["diagnosed_diabetes"]
diabetes_data = full_data.drop("diagnosed_diabetes", axis=1)

# Feature engineering: Add interactions
numeric_cols = diabetes_data.select_dtypes(include=["int64", "float64"]).columns[:8]
for i, col1 in enumerate(numeric_cols[:3]):
    for col2 in numeric_cols[i + 1 : 4]:
        diabetes_data[f"{col1}_x_{col2}"] = diabetes_data[col1] * diabetes_data[col2]

# Split
x_train, x_test, y_train, y_test = train_test_split(
    diabetes_data, labels, test_size=0.1, stratify=labels, random_state=42
)

# Encode categoricals
cat_cols = x_train.select_dtypes(include=["object"]).columns
encoders_copy = {}
for col in cat_cols:
    le = LabelEncoder()
    x_train[col] = le.fit_transform(x_train[col].astype(str))
    x_test[col] = le.transform(x_test[col].astype(str))
    
    encoders_copy[col] = le

x_train = x_train.fillna(x_train.median())
x_test = x_test.fillna(x_train.median())

# Calculate class weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Define base models
base_models = [
    (
        "xgb",
        xgb.XGBClassifier(
            n_estimators=700,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.5,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        ),
    ),
    (
        "rf",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
    (
        "gb",
        GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            random_state=42,
        ),
    ),
]

# Stacking ensemble
print("Training stacking ensemble...")
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
    cv=5,
    n_jobs=-1,
)

stacking.fit(x_train, y_train)

# Get probabilities and optimize threshold
y_pred_proba = stacking.predict_proba(x_test)[:, 1]

thresholds = np.arange(0.35, 0.65, 0.01)
f1_scores = []

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    f1_scores.append(f1_score(y_test, y_pred, average="macro"))

optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"\nOptimal threshold: {optimal_threshold:.3f}")

# Final predictions
predictions = (y_pred_proba >= optimal_threshold).astype(int)

print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.6f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Feature importance from XGBoost
xgb_model = base_models[0][1]
feature_importance = pd.DataFrame(
    {"feature": x_train.columns, "importance": xgb_model.feature_importances_}
).sort_values("importance", ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))


cat_cols = test_data_imputed.select_dtypes(include=["object"]).columns
for col in cat_cols:
    test_data_imputed[col] = encoders_copy[col].transform(test_data_imputed[col].astype(str))

test_data_imputed = test_data_imputed.fillna(x_train.median())

# Get probabilities and optimize threshold
y_pred_proba = stacking.predict_proba(test_data_imputed)[:, 1]

thresholds = np.arange(0.35, 0.65, 0.01)
f1_scores = []

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    f1_scores.append(f1_score(y_test, y_pred, average="macro"))

optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"\nOptimal threshold: {optimal_threshold:.3f}")

# Final predictions
predictions = (y_pred_proba >= optimal_threshold).astype(int)

submission_csv = pd.DataFrame()
submission_csv['id'] = test_data_imputed['id']
submission_csv['diagnosed_diabetes'] = predictions

submission_csv.to_csv('submission.csv', index=False)