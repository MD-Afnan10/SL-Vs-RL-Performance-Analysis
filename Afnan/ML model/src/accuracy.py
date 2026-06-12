import numpy as np
import tensorflow as tf
import joblib

# ======================
# LOAD MODEL
# ======================

dqn = tf.keras.models.load_model(
    "Models/dqn_model.h5",
    compile=False
)

# Load scaler used during training
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

y = np.load(
    "extracted_features/extracted_test/labels.npy"
)

# Apply same normalization used in training
X = scaler.transform(X)

num_samples = len(X)

correct = 0

# ======================
# TEST LOOP
# ======================

for i in range(num_samples):

    state = X[i].reshape(1, -1)

    q_values = dqn.predict(
        state,
        verbose=0
    )

    predicted_class = np.argmax(
        q_values
    )

    if predicted_class == y[i]:
        correct += 1

print("Testing complete ✔")

# ======================
# FINAL ACCURACY
# ======================

accuracy = correct / num_samples

print("\n======================")
print("TEST RESULTS")
print("======================")
print("Total samples:", num_samples)
print("Correct predictions:", correct)
print("Accuracy:", round(accuracy * 100, 2), "%")