import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
dataset = pd.read_csv(r"data/diabetes_dataset.csv")
train_data = pd.read_csv(r"data/train.csv")
test_data = pd.read_csv(r"data/test.csv")

full_data = pd.concat([train_data, dataset], join="inner", ignore_index=True)

labels = full_data["diagnosed_diabetes"]
diabetes_data = full_data.drop("diagnosed_diabetes", axis=1)

# Split first
x_train, x_test, y_train, y_test = train_test_split(
    diabetes_data,
    labels,
    shuffle=True,
    test_size=0.1,
    stratify=labels,
    random_state=42,
)

# One-Hot Encoding (fit on train, transform both)
cat_cols = x_train.select_dtypes(include=["object"]).columns

# Use pandas get_dummies with drop_first to avoid multicollinearity
x_train_encoded = pd.get_dummies(x_train, columns=cat_cols, drop_first=True)
x_test_encoded = pd.get_dummies(x_test, columns=cat_cols, drop_first=True)

# Ensure both have same columns (important!)
x_test_encoded = x_test_encoded.reindex(columns=x_train_encoded.columns, fill_value=0)

# Scale numerical features (CRITICAL for neural networks)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_encoded)
x_test_scaled = scaler.transform(x_test_encoded)

# Calculate class weights for imbalanced data
class_weight = {
    0: len(y_train) / (2 * (y_train == 0).sum()),
    1: len(y_train) / (2 * (y_train == 1).sum()),
}
print(f"Class weights: {class_weight}")

# Build FNN model
model = keras.Sequential(
    [
        layers.Input(shape=(x_train_scaled.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Compile with class weights handled during training
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)

print(model.summary())

# Early stopping callback
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)

# Train
history = model.fit(
    x_train_scaled,
    y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=256,
    class_weight=class_weight,  # Handle imbalance
    callbacks=[early_stop],
    verbose=1,
)

# Predict
y_pred_proba = model.predict(x_test_scaled).flatten()
predictions = (y_pred_proba >= 0.5).astype(int)

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy:.6f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Optional: Threshold tuning
from sklearn.metrics import f1_score

thresholds = np.arange(0.3, 0.7, 0.01)
f1_scores = []

for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_threshold, average="macro"))

optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"\nOptimal threshold: {optimal_threshold:.3f}")

predictions_tuned = (y_pred_proba >= optimal_threshold).astype(int)
print(f"Accuracy with tuned threshold: {accuracy_score(y_test, predictions_tuned):.6f}")
print("\nClassification Report (tuned):")
print(classification_report(y_test, predictions_tuned))
