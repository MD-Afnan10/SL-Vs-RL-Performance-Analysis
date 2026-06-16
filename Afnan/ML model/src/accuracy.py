import numpy as np
import tensorflow as tf
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ======================
# LOAD MODEL
# ======================

dqn = tf.keras.models.load_model(
    "Models/dqn_model.h5",
    compile=False
)

scaler = joblib.load(
    "Models/scaler.pkl"
)

print("Model loaded ✔")

# ======================
# LOAD TEST DATA
# ======================

X = np.load(
    "extracted_features/extracted_test/features.npy"
).astype(np.float32)

y_true = np.load(
    "extracted_features/extracted_test/labels.npy"
)

# Apply same normalization used in training
X = scaler.transform(X)

# ======================
# PREDICTIONS
# ======================

y_pred = []

for i in range(len(X)):

    state = X[i].reshape(1, -1)

    q_values = dqn.predict(
        state,
        verbose=0
    )

    predicted_class = np.argmax(
        q_values
    )

    y_pred.append(predicted_class)

y_pred = np.array(y_pred)

print("Testing complete ✔")

# ======================
# ACCURACY
# ======================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n======================")
print("TEST RESULTS")
print("======================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

# ======================
# CLASSIFICATION REPORT
# ======================

print("\n======================")
print("PRECISION / RECALL / F1")
print("======================")

print(
    classification_report(
        y_true,
        y_pred,
        digits=4
    )
)

# ======================
# CONFUSION MATRIX
# ======================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n======================")
print("CONFUSION MATRIX")
print("======================")

print(cm)