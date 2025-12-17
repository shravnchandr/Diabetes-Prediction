import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve

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

dataset.drop(extra_features, axis=1, inplace=True)
full_data = pd.concat([train_data, dataset], ignore_index=True)

labels = full_data["diagnosed_diabetes"]
diabetes_data = full_data.drop("diagnosed_diabetes", axis=1)

diabetes_data.drop("id", axis=1, inplace=True)


cat_cols = diabetes_data.select_dtypes(include=["object"]).columns
label_encoders = {}

# Encode categorical on FULL training data
for col in cat_cols:
    le = LabelEncoder()
    diabetes_data[col] = diabetes_data[col].fillna("missing").astype(str)
    diabetes_data[col] = le.fit_transform(diabetes_data[col])
    label_encoders[col] = le

# Handle test data (KEEP ID for submission!)
test_submission_ids = test_data["id"].copy()
test_data.drop("id", axis=1, inplace=True)

for col in cat_cols:
    if col in test_data.columns:
        le = label_encoders[col]
        test_data[col] = test_data[col].fillna("missing").astype(str)
        test_data[col] = test_data[col].apply(
            lambda x: x if x in le.classes_ else "missing"
        )
        test_data[col] = le.transform(test_data[col])

# Impute on FULL training data
numeric_cols = diabetes_data.select_dtypes(include=["int64", "float64"]).columns
imputer = SimpleImputer(strategy="median")
diabetes_data[numeric_cols] = imputer.fit_transform(diabetes_data[numeric_cols])
test_data[numeric_cols] = imputer.transform(test_data[numeric_cols])

print(f"Full training set: {diabetes_data.shape}")
print(f"Test set: {test_data.shape}")
print(f"Final feature count: {len(diabetes_data.columns)}")


def train_cv_ensemble_auc(X, y, X_test, seeds=[42, 123, 456], n_folds=5):
    """Train optimized for ROC AUC"""
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
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

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
                scale_pos_weight=(y_train_fold == 0).sum() / (y_train_fold == 1).sum(),
                early_stopping_rounds=50,
                random_state=seed,
                eval_metric="auc",  # AUC for early stopping
            )

            model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False,
            )

            oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]
            test_preds += model.predict_proba(X_test)[:, 1] / n_folds

            fold_auc = roc_auc_score(y_val_fold, oof_preds[val_idx])
            print(f"  Fold {fold+1}: AUC = {fold_auc:.6f}")

        all_oof.append(oof_preds)
        all_test.append(test_preds)

        seed_auc = roc_auc_score(y, oof_preds)
        print(f"Seed {seed} OOF AUC: {seed_auc:.6f}")

    final_oof = np.mean(all_oof, axis=0)
    final_test = np.mean(all_test, axis=0)

    final_auc = roc_auc_score(y, final_oof)
    print(f"\n{'='*60}")
    print(f"Final Ensemble OOF AUC: {final_auc:.6f}")
    print(f"{'='*60}")

    return final_oof, final_test


print("\nTraining CV ensemble optimized for AUC...")
oof_predictions, test_predictions = train_cv_ensemble_auc(
    diabetes_data, labels, test_data, seeds=[42, 123, 456, 789, 2024]
)

oof_auc = roc_auc_score(labels, oof_predictions)
print(f"OOF ROC AUC: {oof_auc:.6f}")

submission = pd.DataFrame(
    {
        "id": test_submission_ids,
        "diagnosed_diabetes": test_predictions,  # Probabilities, NOT binary!
    }
)

submission.to_csv("submission.csv", index=False)
