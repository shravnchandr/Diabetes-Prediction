# diabetes-prediction-v4-stacking.py

import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
SEED = 42
N_FOLDS = 7  # Increased folds for more robust OOF predictions
N_CLUSTERS = 7

# Set competitive hyperparameters (Lower LR, Higher Estimators)
XGB_PARAMS = {
    "n_estimators": 3000,
    "max_depth": 6,
    "learning_rate": 0.015,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "eval_metric": "auc",
    "early_stopping_rounds": 100,
    "random_state": SEED,
    "n_jobs": -1,
}
LGB_PARAMS = {
    "n_estimators": 3000,
    "num_leaves": 31,
    "max_depth": 8,
    "learning_rate": 0.015,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "metric": "auc",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}
CAT_PARAMS = {
    "iterations": 3000,
    "depth": 6,
    "learning_rate": 0.02,
    "eval_metric": "AUC",
    "random_seed": SEED,
    "verbose": False,
    "allow_writing_files": False,
    "early_stopping_rounds": 100,
}

# ==================== 1. DATA LOADING ====================
print("Loading data...")
dataset = pd.read_csv(
    r"/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv"
)
train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e12/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e12/test.csv")

# Save IDs for submission
test_ids = test_data["id"].copy()

# ==================== 2. MODEL-BASED IMPUTATION (Transfer Learning) ====================
# Keeping the robust imputation from the previous version
extra_features = [
    col for col in dataset.columns if col not in train_data.columns and col != "id"
]
common_features = [
    col
    for col in dataset.columns
    if col in train_data.columns and col != "diagnosed_diabetes"
]

imputation_models = {}

print("-" * 10 + " Training Imputation Models " + "-" * 10)
for feature in extra_features:
    if feature not in dataset.columns:
        continue
    mask = dataset[feature].notna()
    if mask.sum() < 50:
        continue
    X = dataset.loc[mask, common_features].copy()
    y = dataset.loc[mask, feature]
    X_encoded = X.copy()
    cat_cols = X_encoded.select_dtypes(include=["object"]).columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        X_encoded[col] = X_encoded[col].fillna("MISSING").astype(str)
        X_encoded[col] = le.fit_transform(X_encoded[col])
        encoders[col] = le

    is_categorical = y.dtype == "object" or y.nunique() < 20

    if is_categorical:
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y.astype(str))
        model = RandomForestClassifier(
            n_estimators=50, max_depth=8, n_jobs=-1, random_state=SEED
        )
        model.fit(X_encoded, y_encoded)
        encoders["target"] = le_target
    else:
        model = RandomForestRegressor(
            n_estimators=50, max_depth=8, n_jobs=-1, random_state=SEED
        )
        model.fit(X_encoded, y)

    imputation_models[feature] = {
        "model": model,
        "encoders": encoders,
        "is_categorical": is_categorical,
    }


def safe_impute(data, models, common_features):
    result = data.copy()
    for feature, model_dict in models.items():
        model = model_dict["model"]
        encoders = model_dict["encoders"]
        X = data[common_features].copy()

        for col, encoder in encoders.items():
            if col == "target" or col not in X.columns:
                continue
            X[col] = X[col].fillna("MISSING").astype(str)
            known_labels = set(encoder.classes_)
            X[col] = X[col].apply(
                lambda x: x if x in known_labels else encoder.classes_[0]
            )
            X[col] = encoder.transform(X[col])

        preds = model.predict(X)
        if model_dict["is_categorical"] and "target" in encoders:
            preds = encoders["target"].inverse_transform(preds.astype(int))
        result[feature] = preds
    return result


print("Imputing Train Data...")
train_data_imputed = safe_impute(train_data, imputation_models, common_features)
print("Imputing Test Data...")
test_data_imputed = safe_impute(test_data, imputation_models, common_features)
print("Imputing Dataset Data...")
dataset_imputed = safe_impute(dataset, imputation_models, common_features)


# ==================== 3. AGGRESSIVE FEATURE ENGINEERING ====================
def feature_engineering(data):
    df = data.copy()

    # Log Transforms
    for col in ["hba1c", "glucose_postprandial", "glucose_fasting", "insulin_level"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    # Medical Interaction Ratios
    if "glucose_postprandial" in df.columns and "glucose_fasting" in df.columns:
        df["glucose_diff"] = df["glucose_postprandial"] - df["glucose_fasting"]
        df["glucose_ratio"] = df["glucose_postprandial"] / (df["glucose_fasting"] + 1.0)

    if "bmi" in df.columns and "age" in df.columns:
        df["bmi_age_interaction"] = df["bmi"] * df["age"]
        df["age_squared"] = df["age"] ** 2  # New aggressive feature
        df["bmi_squared"] = df["bmi"] ** 2  # New aggressive feature

    if "systolic_bp" in df.columns and "diastolic_bp" in df.columns:
        df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]

    # Hypertension interactions
    if "high_bp" in df.columns and "high_cholesterol" in df.columns:
        df["bp_chol_risk"] = df["high_bp"] * df["high_cholesterol"]

    # Binning
    if "age" in df.columns:
        df["age_group"] = pd.cut(df["age"], bins=5, labels=False, include_lowest=True)

    if "bmi" in df.columns:
        df["bmi_cat"] = pd.cut(
            df["bmi"],
            bins=[-1, 18.5, 25, 30, 1000],
            labels=["Under", "Normal", "Over", "Obese"],
            include_lowest=True,
        )

    return df


print("Applying Feature Engineering...")
X_train_fe = feature_engineering(train_data_imputed)
X_test_fe = feature_engineering(test_data_imputed)
X_dataset_fe = feature_engineering(dataset_imputed)

# Clean up IDs
target_col = "diagnosed_diabetes"
if "id" in X_train_fe.columns:
    X_train_fe.drop("id", axis=1, inplace=True)
if "id" in X_test_fe.columns:
    X_test_fe.drop("id", axis=1, inplace=True)
if "id" in X_dataset_fe.columns:
    X_dataset_fe.drop("id", axis=1, inplace=True)

# Define columns for scaling and clustering
numeric_cols = X_train_fe.select_dtypes(include=["int64", "float64"]).columns
numeric_cols = [c for c in numeric_cols if c not in [target_col, "is_generated"]]

# Standard Scaling (Crucial for Stacking and Clustering)
scaler = StandardScaler()
X_train_fe[numeric_cols] = scaler.fit_transform(
    X_train_fe[numeric_cols].fillna(X_train_fe[numeric_cols].mean())
)
X_test_fe[numeric_cols] = scaler.transform(
    X_test_fe[numeric_cols].fillna(X_test_fe[numeric_cols].mean())
)
X_dataset_fe[numeric_cols] = scaler.transform(
    X_dataset_fe[numeric_cols].fillna(X_dataset_fe[numeric_cols].mean())
)

# KMeans Clustering Features
kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED, batch_size=256)
X_train_fe["cluster_id"] = kmeans.fit_predict(X_train_fe[numeric_cols])
X_test_fe["cluster_id"] = kmeans.predict(X_test_fe[numeric_cols])

# 4. Data Blending and Preparation (Using user's strategy)
X_train_fe[target_col] = train_data[target_col]
X_dataset_fe[target_col] = dataset[target_col]

# Add the 'is_generated' flag (as requested by user)
X_train_fe["is_generated"] = 1
X_dataset_fe["is_generated"] = 0
X_test_fe["is_generated"] = 1

# Combine Train and External Dataset
y = X_train_fe[target_col]
X = X_train_fe.drop(target_col, axis=1)

y_dataset = X_dataset_fe[target_col]
X_dataset = X_dataset_fe.drop(target_col, axis=1)

X_combined = pd.concat([X, X_dataset], ignore_index=True)
y_combined = pd.concat([y, y_dataset], ignore_index=True)

# Re-align test columns
X_test = X_test_fe[X_combined.columns]

# ==================== 5. PREPROCESSING FOR ENSEMBLE ====================
# Identify categorical columns (low unique count, not binary)
cat_cols = (
    X_combined.select_dtypes(include=["object", "category", "int64"])
    .nunique()[
        X_combined.select_dtypes(include=["object", "category", "int64"]).nunique() < 20
    ]
    .index.tolist()
)
binary_cols = [
    c
    for c in X_combined.columns
    if X_combined[c].nunique() == 2 and X_combined[c].isin([0, 1]).all()
]
cat_cols = [c for c in cat_cols if c not in binary_cols]


# Set up dataframes for each model:
# LGB/Cat: Use 'category' dtype
X_lgb = X_combined.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

# XGB: Use Label Encoding
X_xgb = X_combined.copy()
X_test_xgb = X_test.copy()
for col in cat_cols:
    le = LabelEncoder()
    full_vals = pd.concat([X_xgb[col], X_test_xgb[col]]).astype(str)
    le.fit(full_vals)
    X_xgb[col] = le.transform(X_xgb[col].astype(str))
    X_test_xgb[col] = le.transform(X_test_xgb[col].astype(str))

# Isolate the original training indices for OOF calculation
N_TRAIN = len(train_data)
y_final = y_combined.iloc[:N_TRAIN]


# ==================== 6. STACKING ENSEMBLE TRAINING LOOP ====================
def train_stacking_ensemble(
    X_xgb, X_lgb, y_combined, X_test_xgb, X_test_lgb, cat_cols, N_TRAIN
):

    # OOF predictions for the meta-learner (only on the original train set)
    oof_xgb = np.zeros(N_TRAIN)
    oof_lgb = np.zeros(N_TRAIN)
    oof_cat = np.zeros(N_TRAIN)

    # Test predictions for the final submission
    test_preds_xgb = np.zeros(len(X_test_xgb))
    test_preds_lgb = np.zeros(len(X_test_xgb))
    test_preds_cat = np.zeros(len(X_test_xgb))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    print(f"\nStarting {N_FOLDS}-Fold Stacking Training...")

    for fold, (train_idx_oof, val_idx_oof) in enumerate(
        skf.split(X_xgb.iloc[:N_TRAIN], y_combined.iloc[:N_TRAIN])
    ):
        print(f"\n--- Fold {fold+1} ---")

        # Training indices: Include the OOF train split + all external data
        train_idx = np.concatenate([train_idx_oof, np.arange(N_TRAIN, len(X_xgb))])

        # Base Model Splits
        X_tr_xgb, X_val_xgb = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx_oof]
        y_tr, y_val = y_combined.iloc[train_idx], y_combined.iloc[val_idx_oof]
        X_tr_lgb, X_val_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx_oof]

        # ---------------- BASE MODELS (Level 0) ----------------

        # 1. XGBOOST
        xgb = XGBClassifier(**XGB_PARAMS)
        xgb.fit(X_tr_xgb, y_tr, eval_set=[(X_val_xgb, y_val)], verbose=False)
        oof_xgb[val_idx_oof] = xgb.predict_proba(X_val_xgb)[:, 1]
        test_preds_xgb += xgb.predict_proba(X_test_xgb)[:, 1] / N_FOLDS

        # 2. LIGHTGBM
        lgbm = lgb.LGBMClassifier(**LGB_PARAMS)
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
        lgbm.fit(X_tr_lgb, y_tr, eval_set=[(X_val_lgb, y_val)], callbacks=callbacks)
        oof_lgb[val_idx_oof] = lgbm.predict_proba(X_val_lgb)[:, 1]
        test_preds_lgb += lgbm.predict_proba(X_test_lgb)[:, 1] / N_FOLDS

        # 3. CATBOOST
        cat = CatBoostClassifier(**CAT_PARAMS, cat_features=cat_cols)
        cat.fit(X_tr_lgb, y_tr, eval_set=(X_val_lgb, y_val))
        oof_cat[val_idx_oof] = cat.predict_proba(X_val_lgb)[:, 1]
        test_preds_cat += cat.predict_proba(X_test_lgb)[:, 1] / N_FOLDS

        fold_blend = (
            oof_xgb[val_idx_oof] + oof_lgb[val_idx_oof] + oof_cat[val_idx_oof]
        ) / 3
        print(f"  Base AUC (Avg): {roc_auc_score(y_val, fold_blend):.5f}")

    # ---------------- STACKER MODEL (Level 1) ----------------
    print("\n--- Training Stacker (Level 1) ---")

    # Create the OOF Meta-Features (Level 1 Training Data)
    X_meta = pd.DataFrame({"xgb": oof_xgb, "lgb": oof_lgb, "cat": oof_cat})

    # The Stacker (Logistic Regression with regularization) learns the optimal weights
    meta_model = LogisticRegression(
        C=0.1, solver="lbfgs", penalty="l2", random_state=SEED, max_iter=1000
    )
    meta_model.fit(X_meta, y_combined.iloc[:N_TRAIN])

    # Final OOF Prediction (used for final AUC check)
    final_oof_preds = meta_model.predict_proba(X_meta)[:, 1]

    # Create the Test Meta-Features (Level 1 Prediction Data)
    X_test_meta = pd.DataFrame(
        {"xgb": test_preds_xgb, "lgb": test_preds_lgb, "cat": test_preds_cat}
    )

    # Final Test Prediction (used for submission)
    final_test_preds = meta_model.predict_proba(X_test_meta)[:, 1]

    print(
        f"  Stacker Weights: {meta_model.coef_[0].round(4)}"
    )  # Should show the learned combination
    print(f"  Stacker Intercept (Bias): {meta_model.intercept_[0]:.4f}")

    return final_oof_preds, final_test_preds


# Run Training
oof_predictions, test_predictions = train_stacking_ensemble(
    X_xgb, X_lgb, y_combined, X_test_xgb, X_test_lgb, cat_cols, N_TRAIN
)

# ==================== 7. EVALUATION & SUBMISSION ====================
final_auc = roc_auc_score(y_final, oof_predictions)
print(f"\n{'='*40}")
print(f"FINAL OOF AUC SCORE (Stacking): {final_auc:.6f}")
print(f"{'='*40}")

# Create Submission
submission = pd.DataFrame({"id": test_ids, "diagnosed_diabetes": test_predictions})

submission.to_csv("submission_stacking_optimized.csv", index=False)
print("✓ Submission saved to 'submission_stacking_optimized.csv'")
print(submission.head())
