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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
SEED = 42
N_FOLDS = 5
N_CLUSTERS = 7  # For feature engineering

# ==================== 1. DATA LOADING ====================
print("Loading data...")
dataset = pd.read_csv(
    r"/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv"
)
train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e12/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e12/test.csv")

# Save IDs for submission
test_ids = test_data["id"].copy()

# ==================== 2. ROBUST MODEL-BASED IMPUTATION ====================
# We keep your transfer learning logic but make the encoding safer
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
        continue  # Skip if too little data

    X = dataset.loc[mask, common_features].copy()
    y = dataset.loc[mask, feature]

    # Robust Encoding for Imputation Training
    X_encoded = X.copy()
    cat_cols = X_encoded.select_dtypes(include=["object"]).columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        # Convert to string and handle missing
        X_encoded[col] = X_encoded[col].fillna("MISSING").astype(str)
        X_encoded[col] = le.fit_transform(X_encoded[col])
        encoders[col] = le

    # Determine type of model needed
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
    print(f"✓ Trained imputer for: {feature}")


def safe_impute(data, models, common_features):
    result = data.copy()
    for feature, model_dict in models.items():
        model = model_dict["model"]
        encoders = model_dict["encoders"]

        X = data[common_features].copy()

        # Robust Encode
        for col, encoder in encoders.items():
            if col == "target" or col not in X.columns:
                continue

            # Map known classes, fill unknown with a special value or mode
            X[col] = X[col].fillna("MISSING").astype(str)

            # Safe transform: unknown labels get assigned to the most frequent class (index 0 usually safe enough here or strictly handle)
            # A safer way using numpy:
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


# ==================== 3. ADVANCED FEATURE ENGINEERING ====================
def feature_engineering(data, is_train=True):
    df = data.copy()

    # 1. Log Transforms (Skewed data)
    for col in ["hba1c", "glucose_postprandial", "glucose_fasting", "insulin_level"]:
        if col in df.columns:
            # Add +1 to avoid log(0)
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    # 2. Medical Interaction Ratios (High Value)
    if "glucose_postprandial" in df.columns and "glucose_fasting" in df.columns:
        df["glucose_diff"] = df["glucose_postprandial"] - df["glucose_fasting"]
        df["glucose_ratio"] = df["glucose_postprandial"] / (df["glucose_fasting"] + 1.0)

    if "bmi" in df.columns and "age" in df.columns:
        # BMI impact increases with age generally
        df["bmi_age_interaction"] = df["bmi"] * df["age"]

    if "systolic_bp" in df.columns and "diastolic_bp" in df.columns:
        df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]

    # 3. Binning (Helps trees isolate groups)
    if "age" in df.columns:
        df["age_group"] = pd.cut(df["age"], bins=5, labels=False)

    if "bmi" in df.columns:
        # Standard BMI categories: Under(<18.5), Normal(18.5-25), Over(25-30), Obese(>30)
        df["bmi_cat"] = pd.cut(
            df["bmi"],
            bins=[-1, 18.5, 25, 30, 100],
            labels=["Under", "Normal", "Over", "Obese"],
        )

    return df


# Combine for consistent FE
full_df = pd.concat([train_data_imputed, dataset], ignore_index=True)
if "diagnosed_diabetes" not in train_data_imputed.columns:
    # Handle case where dataset has target but train_data needs it dropped temporarily if needed
    pass

# Apply FE
print("Applying Feature Engineering...")
# We process train/test separately to avoid leakage, or carefully together.
# Here we apply to train and test independently.
X_train_fe = feature_engineering(train_data_imputed)
X_test_fe = feature_engineering(test_data_imputed)
X_dataset_fe = feature_engineering(dataset)

X_train_fe["is_generated"] = 1
X_test_fe["is_generated"] = 1
X_dataset_fe["is_generated"] = 0

X_train_fe = pd.concat([X_train_fe, X_dataset_fe], ignore_index=True)

# Optional: Add KMeans Clustering Features
# (Fits on numeric columns to find "patient prototypes")
numeric_cols = X_train_fe.select_dtypes(include=["int64", "float64"]).columns
numeric_cols = [c for c in numeric_cols if c not in ["id", "diagnosed_diabetes"]]

# Simple imputation for clustering
kmeans_imputer = SimpleImputer(strategy="median")
X_clus = kmeans_imputer.fit_transform(X_train_fe[numeric_cols])
X_test_clus = kmeans_imputer.transform(X_test_fe[numeric_cols])

kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED, batch_size=256)
X_train_fe["cluster_id"] = kmeans.fit_predict(X_clus)
X_test_fe["cluster_id"] = kmeans.predict(X_test_clus)

# Clean up columns
target_col = "diagnosed_diabetes"
if "id" in X_train_fe.columns:
    X_train_fe.drop("id", axis=1, inplace=True)
if "id" in X_test_fe.columns:
    X_test_fe.drop("id", axis=1, inplace=True)

y = X_train_fe[target_col]
X = X_train_fe.drop(target_col, axis=1)

# Align test columns
X_test = X_test_fe[X.columns]

# ==================== 4. PREPROCESSING FOR ENSEMBLE ====================
# Identify categorical columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
# Also add the new binned/cluster columns if they are treated as cats
if "cluster_id" in X.columns:
    cat_cols.append("cluster_id")
if "age_group" in X.columns:
    cat_cols.append("age_group")

# For LightGBM: Convert to 'category' dtype
# For CatBoost: Keep track of names
# For XGBoost: Enable categorical support or Encode

X_lgb = X.copy()
X_test_lgb = X_test.copy()

for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

# For XGBoost (Needs numeric or specialized encoding)
# We will use simple Ordinal/Label encoding for XGB to be safe
X_xgb = X.copy()
X_test_xgb = X_test.copy()
for col in cat_cols:
    le = LabelEncoder()
    # Handle full set of values
    full_vals = pd.concat([X_xgb[col], X_test_xgb[col]]).astype(str)
    le.fit(full_vals)
    X_xgb[col] = le.transform(X_xgb[col].astype(str))
    X_test_xgb[col] = le.transform(X_test_xgb[col].astype(str))


# ==================== 5. ENSEMBLE TRAINING LOOP ====================
def train_ensemble(X_xgb, X_lgb, y, X_test_xgb, X_test_lgb, cat_cols):

    # Store predictions
    oof_preds = np.zeros(len(X_xgb))
    test_preds = np.zeros(len(X_test_xgb))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Weights for the blend (can be tuned)
    w_xgb = 0.35
    w_lgb = 0.35
    w_cat = 0.30

    print(f"\nStarting {N_FOLDS}-Fold Ensemble Training...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_xgb, y)):
        print(f"\n--- Fold {fold+1} ---")

        # Split Data
        # XGB
        X_tr_xgb, X_val_xgb = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # LGB
        X_tr_lgb, X_val_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]

        # CatBoost (can use lgb data or raw, we use lgb data but passed as cat features)
        X_tr_cat, X_val_cat = X_tr_lgb, X_val_lgb

        # ---------------- MODEL 1: XGBOOST ----------------
        xgb = XGBClassifier(
            n_estimators=2000,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            eval_metric="auc",
            early_stopping_rounds=100,
        )
        xgb.fit(X_tr_xgb, y_tr, eval_set=[(X_val_xgb, y_val)], verbose=False)
        p_xgb = xgb.predict_proba(X_val_xgb)[:, 1]
        t_xgb = xgb.predict_proba(X_test_xgb)[:, 1]
        print(f"  XGB AUC: {roc_auc_score(y_val, p_xgb):.5f}")

        # ---------------- MODEL 2: LIGHTGBM ----------------
        lgbm = lgb.LGBMClassifier(
            n_estimators=2000,
            num_leaves=31,
            max_depth=8,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            metric="auc",
            verbose=-1,
        )
        # Callbacks for early stopping
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
        lgbm.fit(X_tr_lgb, y_tr, eval_set=[(X_val_lgb, y_val)], callbacks=callbacks)
        p_lgb = lgbm.predict_proba(X_val_lgb)[:, 1]
        t_lgb = lgbm.predict_proba(X_test_lgb)[:, 1]
        print(f"  LGB AUC: {roc_auc_score(y_val, p_lgb):.5f}")

        # ---------------- MODEL 3: CATBOOST ----------------
        cat = CatBoostClassifier(
            iterations=2000,
            depth=6,
            learning_rate=0.03,
            eval_metric="AUC",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
            cat_features=cat_cols,  # Uses native handling
        )
        cat.fit(X_tr_cat, y_tr, eval_set=(X_val_cat, y_val), early_stopping_rounds=100)
        p_cat = cat.predict_proba(X_val_cat)[:, 1]
        t_cat = cat.predict_proba(X_test_lgb)[:, 1]  # Use DF with correct dtypes
        print(f"  Cat AUC: {roc_auc_score(y_val, p_cat):.5f}")

        # ---------------- BLEND ----------------
        # Weighted Average
        blend_oof = (w_xgb * p_xgb) + (w_lgb * p_lgb) + (w_cat * p_cat)
        oof_preds[val_idx] = blend_oof

        blend_test = (w_xgb * t_xgb) + (w_lgb * t_lgb) + (w_cat * t_cat)
        test_preds += blend_test / N_FOLDS

        print(f"  >> BLEND AUC: {roc_auc_score(y_val, blend_oof):.5f}")

    return oof_preds, test_preds


# Run Training
oof_predictions, test_predictions = train_ensemble(
    X_xgb, X_lgb, y, X_test_xgb, X_test_lgb, cat_cols
)

# ==================== 6. EVALUATION & SUBMISSION ====================
final_auc = roc_auc_score(y, oof_predictions)
print(f"\n{'='*40}")
print(f"FINAL OOF AUC SCORE: {final_auc:.6f}")
print(f"{'='*40}")

# Plot ROC
fpr, tpr, _ = roc_curve(y, oof_predictions)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Ensemble (AUC = {final_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.title("ROC Curve - Ensemble Model")
plt.legend()
plt.show()

# Create Submission
submission = pd.DataFrame({"id": test_ids, "diagnosed_diabetes": test_predictions})

submission.to_csv("submission_optimized.csv", index=False)
print("✓ Submission saved to 'submission_optimized.csv'")
print(submission.head())
