import pandas as pd
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv(r"data/diabetes_dataset.csv")
train_data = pd.read_csv(r"data/train.csv")
test_data = pd.read_csv(r"data/test.csv")

train_data = pd.concat([train_data, dataset], join="inner", ignore_index=True)
cat_cols = train_data.select_dtypes(include=['object']).columns

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le

print(train_data.info())