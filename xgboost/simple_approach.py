import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

dataset = pd.read_csv(
    r"/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv"
)
train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e12/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e12/test.csv")

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


print("-" * 10 + "Imputing Train Data" + "-" * 10)
train_data_imputed = impute_data(train_data, imputation_models, common_features)
print("-" * 10 + "Imputing Test Data" + "-" * 10)
test_data_imputed = impute_data(test_data, imputation_models, common_features)


def create_all_features(data):
    data = data.copy()

    # Log transforms for skewed features
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if (data[col] > 0).all():  # Only if all positive
            data[f"{col}_log"] = np.log1p(data[col])

    # Frequency encoding for categorical
    cat_cols = data.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        freq = data[col].value_counts(normalize=True)
        data[f"{col}_freq"] = data[col].map(freq)

    # Group statistics
    if "age" in data.columns and "bmi" in data.columns:
        data["age_group"] = pd.cut(data["age"], bins=[0, 30, 45, 60, 100], labels=False)
        bmi_by_age = data.groupby("age_group")["bmi"].transform("mean")
        data["bmi_vs_age_avg"] = data["bmi"] - bmi_by_age

    # Medical ratios
    if "glucose_postprandial" in data.columns and "glucose_fasting" in data.columns:
        data["glucose_ratio"] = data["glucose_postprandial"] / (
            data["glucose_fasting"] + 1
        )
        data["glucose_diff"] = data["glucose_postprandial"] - data["glucose_fasting"]

    if "insulin_level" in data.columns and "glucose_fasting" in data.columns:
        # HOMA-IR
        data["insulin_resistance"] = (
            data["insulin_level"] * data["glucose_fasting"]
        ) / 405

    # Interactions
    interaction_pairs = [("age", "bmi"), ("glucose_fasting", "bmi")]
    for col1, col2 in interaction_pairs:
        if col1 in data.columns and col2 in data.columns:
            data[f"{col1}_x_{col2}"] = data[col1] * data[col2]

    # Polynomials for key features
    for feature in ["bmi", "age", "glucose_fasting"]:
        if feature in data.columns:
            data[f"{feature}_squared"] = data[feature] ** 2

    return data


full_data = pd.concat([train_data_imputed, dataset], ignore_index=True)

full_data_fe = create_all_features(full_data)
test_data_imputed_fe = create_all_features(test_data_imputed)

labels = full_data_fe["diagnosed_diabetes"]
diabetes_data = full_data_fe.drop("diagnosed_diabetes", axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    diabetes_data, labels, test_size=0.1, stratify=labels, random_state=42
)

# Label encoding
cat_cols = x_train.select_dtypes(include=["object"]).columns
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    x_train[col] = x_train[col].fillna("missing").astype(str)
    x_test[col] = x_test[col].fillna("missing").astype(str)

    x_train[col] = le.fit_transform(x_train[col])
    x_test[col] = le.transform(x_test[col])
    label_encoders[col] = le

# Handle test data categorical
for col in cat_cols:
    if col in test_data_imputed_fe.columns:
        le = label_encoders[col]
        test_data_imputed_fe[col] = (
            test_data_imputed_fe[col].fillna("missing").astype(str)
        )
        test_data_imputed_fe[col] = test_data_imputed_fe[col].apply(
            lambda x: x if x in le.classes_ else "missing"
        )
        test_data_imputed_fe[col] = le.transform(test_data_imputed_fe[col])

# Align columns
missing_cols = set(x_train.columns) - set(test_data_imputed_fe.columns)
for col in missing_cols:
    test_data_imputed_fe[col] = 0
test_data_imputed_fe = test_data_imputed_fe[x_train.columns]

# Impute numeric
numeric_cols = x_train.select_dtypes(include=["int64", "float64"]).columns
imputer = SimpleImputer(strategy="median")
x_train[numeric_cols] = imputer.fit_transform(x_train[numeric_cols])
x_test[numeric_cols] = imputer.transform(x_test[numeric_cols])
test_data_imputed_fe[numeric_cols] = imputer.transform(
    test_data_imputed_fe[numeric_cols]
)


def train_cv_ensemble(X, y, X_test, seeds=[42, 123, 456], n_folds=5):
    all_oof = []
    all_test = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            model = XGBClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.5,
                scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
                early_stopping_rounds=50,
                random_state=seed,
                eval_metric="logloss",
            )

            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            test_preds += model.predict_proba(X_test)[:, 1] / n_folds

            fold_f1 = f1_score(
                y_val,
                (model.predict_proba(X_val)[:, 1] >= 0.5).astype(int),
                average="macro",
            )
            print(f"  Fold {fold+1}: F1 = {fold_f1:.4f}")

        all_oof.append(oof_preds)
        all_test.append(test_preds)

        seed_f1 = f1_score(y, (oof_preds >= 0.5).astype(int), average="macro")
        print(f"Seed {seed} OOF F1: {seed_f1:.6f}")

    final_oof = np.mean(all_oof, axis=0)
    final_test = np.mean(all_test, axis=0)

    final_f1 = f1_score(y, (final_oof >= 0.5).astype(int), average="macro")
    print(f"\n{'='*60}")
    print(f"Final Ensemble OOF F1: {final_f1:.6f}")
    print(f"{'='*60}")

    return final_oof, final_test


X_full = pd.concat([x_train, x_test], ignore_index=True)
y_full = pd.concat([y_train, y_test], ignore_index=True)

print("Training CV ensemble on full data...")
oof_predictions, test_predictions = train_cv_ensemble(
    X_full, y_full, test_data_imputed_fe, seeds=[42, 123, 456]  
)


thresholds = np.arange(0.3, 0.7, 0.005)
f1_scores_threshold = [
    f1_score(y_full, (oof_predictions >= t).astype(int), average="macro")
    for t in thresholds
]
optimal_threshold = thresholds[np.argmax(f1_scores_threshold)]
print(f"\nOptimal threshold: {optimal_threshold:.4f}")

final_predictions = (test_predictions >= optimal_threshold).astype(int)

submission = pd.DataFrame({"id": test_ids, "diagnosed_diabetes": final_predictions})
submission.to_csv("submission.csv", index=False)
print(f"Prediction distribution:\n{submission['diagnosed_diabetes'].value_counts()}")
