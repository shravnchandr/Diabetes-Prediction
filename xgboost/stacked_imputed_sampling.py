import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek


dataset = pd.read_csv(
    r"/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv"
)
train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e12/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e12/test.csv")

# Save test IDs before any processing
test_ids = test_data["id"].copy()

extra_features = [
    col for col in dataset.columns if col not in train_data.columns and col != "id"
]

common_features = [
    col
    for col in dataset.columns
    if col in train_data.columns and col != "diagnosed_diabetes"
]

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


def create_features(data):
    """Apply same feature engineering to all datasets"""
    data = data.copy()
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns[:8]
    for i, col1 in enumerate(numeric_cols[:3]):
        for col2 in numeric_cols[i + 1 : 4]:
            if col1 in data.columns and col2 in data.columns:
                data[f"{col1}_x_{col2}"] = data[col1] * data[col2]
    return data


def advanced_features(data):
    """Focus on features that help even tree-based models"""
    data = data.copy()

    if "glucose_fasting" in data.columns and "glucose_postprandial" in data.columns:
        # Clinical significance: how much glucose rises after eating
        data["glucose_spike"] = data["glucose_postprandial"] - data["glucose_fasting"]
        data["glucose_spike_ratio"] = data["glucose_postprandial"] / (
            data["glucose_fasting"] + 1
        )

    if "insulin_level" in data.columns and "glucose_fasting" in data.columns:
        # HOMA-IR: actual clinical metric for insulin resistance
        data["insulin_resistance"] = (
            data["insulin_level"] * data["glucose_fasting"]
        ) / 405

    interaction_pairs = [
        ("age", "bmi"),
        ("age", "glucose_fasting"),
        ("bmi", "glucose_fasting"),
        ("age", "insulin_level"),
    ]

    for col1, col2 in interaction_pairs:
        if col1 in data.columns and col2 in data.columns:
            data[f"{col1}_x_{col2}"] = data[col1] * data[col2]

    # ========== POLYNOMIALS (Non-linear relationships) ==========
    # Only for key features with known non-linear effects
    polynomial_features = ["bmi", "age", "glucose_fasting"]

    for feature in polynomial_features:
        if feature in data.columns:
            data[f"{feature}_squared"] = data[feature] ** 2
            # Square root for potential logarithmic relationships
            data[f"{feature}_sqrt"] = np.sqrt(np.abs(data[feature]))

    if all(col in data.columns for col in ["bmi", "glucose_fasting", "age"]):
        # Normalize to 0-1 range for each component
        from sklearn.preprocessing import MinMaxScaler

        risk_components = data[["bmi", "glucose_fasting", "age"]].copy()
        scaler = MinMaxScaler()
        risk_normalized = scaler.fit_transform(risk_components)

        # Weighted sum based on clinical importance
        data["metabolic_risk"] = (
            risk_normalized[:, 0] * 0.3  # BMI
            + risk_normalized[:, 1] * 0.5  # Glucose (most important)
            + risk_normalized[:, 2] * 0.2  # Age
        )

    return data


full_data = pd.concat([train_data_imputed, dataset], ignore_index=True)
labels = full_data["diagnosed_diabetes"]
diabetes_data = full_data.drop("diagnosed_diabetes", axis=1)

diabetes_data = create_features(diabetes_data)
diabetes_data = advanced_features(diabetes_data)

x_train, x_test, y_train, y_test = train_test_split(
    diabetes_data, labels, test_size=0.1, stratify=labels, random_state=42
)

cat_cols = x_train.select_dtypes(include=["object"]).columns
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    x_train[col] = x_train[col].fillna("missing").astype(str)
    x_test[col] = x_test[col].fillna("missing").astype(str)

    x_train[col] = le.fit_transform(x_train[col])
    x_test[col] = le.transform(x_test[col])
    label_encoders[col] = le

numeric_cols = x_train.select_dtypes(include=["int64", "float64"]).columns
imputer = SimpleImputer(strategy="median")
x_train[numeric_cols] = imputer.fit_transform(x_train[numeric_cols])
x_test[numeric_cols] = imputer.transform(x_test[numeric_cols])

# smote_tomek = SMOTETomek(
#     smote=SMOTE(sampling_strategy=0.8, random_state=42), random_state=42
# )

sm = SMOTE(sampling_strategy=0.8, random_state=42)

x_train, y_train = sm.fit_resample(x_train, y_train)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

base_models = [
    (
        "xgb",
        XGBClassifier(
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
        "catboost",
        CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False,
            auto_class_weights="Balanced",
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
        "hist_gb",
        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
        ),
    ),
]

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=1000, solver="saga"
    ),
    cv=5,
    n_jobs=-1,
)

stacking.fit(x_train, y_train)

# Calibrate probabilities
calibrated_stacking = CalibratedClassifierCV(stacking, method="sigmoid", cv=3)
calibrated_stacking.fit(x_train, y_train)

# Optimize threshold
y_pred_proba = calibrated_stacking.predict_proba(x_test)[:, 1]
thresholds = np.arange(0.3, 0.7, 0.005)

f1_scores = []

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    f1_scores.append(f1_score(y_test, y_pred, average="macro"))

optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold: {optimal_threshold:.3f}")

predictions = (y_pred_proba >= optimal_threshold).astype(int)
print(f"\nValidation Results:")
print(f"Accuracy: {accuracy_score(y_test, predictions):.6f}")
print(classification_report(y_test, predictions))

test_data_imputed = create_features(test_data_imputed)
test_data_imputed = advanced_features(test_data_imputed)

cat_cols = test_data_imputed.select_dtypes(include=["object"]).columns
for col in cat_cols:
    if col in label_encoders:
        le = label_encoders[col]
        test_data_imputed[col] = test_data_imputed[col].fillna("missing").astype(str)
        test_data_imputed[col] = test_data_imputed[col].apply(
            lambda x: x if x in le.classes_ else "missing"
        )
        test_data_imputed[col] = le.transform(test_data_imputed[col])

missing_cols = set(x_train.columns) - set(test_data_imputed.columns)
for col in missing_cols:
    test_data_imputed[col] = 0

test_data_imputed = test_data_imputed[x_train.columns]

numeric_cols = test_data_imputed.select_dtypes(include=["int64", "float64"]).columns
test_data_imputed[numeric_cols] = imputer.transform(test_data_imputed[numeric_cols])

y_pred_proba_test = calibrated_stacking.predict_proba(test_data_imputed)[:, 1]
predictions_test = (y_pred_proba_test >= optimal_threshold).astype(int)

submission_csv = pd.DataFrame({"id": test_ids, "diagnosed_diabetes": predictions_test})

submission_csv.to_csv("submission.csv", index=False)

print(f"\n✓ Submission saved!")
print(f"Total predictions: {len(submission_csv)}")
print(f"\nPrediction distribution:")
print(submission_csv["diagnosed_diabetes"].value_counts())
print(f"\nProportion of positive class: {(predictions_test == 1).mean():.3f}")
