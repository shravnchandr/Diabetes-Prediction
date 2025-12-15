import pandas as pd

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

dataset = pd.read_csv(r"data/diabetes_dataset.csv")
train_data = pd.read_csv(r"data/train.csv")
test_data = pd.read_csv(r"data/test.csv")

full_data = pd.concat([train_data, dataset], join="inner", ignore_index=True)
cat_cols = full_data.select_dtypes(include=["object"]).columns

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    full_data[col] = le.fit_transform(full_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le

labels = full_data["diagnosed_diabetes"]
diabetes_data = full_data.drop("diagnosed_diabetes", axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    diabetes_data,
    labels,
    shuffle=True,
    test_size=0.1,
    stratify=full_data["diagnosed_diabetes"],
    random_state=42,
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    early_stopping_rounds=50,
)

model.fit(x_train, y_train, eval_set=[(x_test, y_test)])
predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))
