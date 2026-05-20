import numpy as np
import tensorflow as tf
import os
from utils import load_feature_extractor, preprocess_image, classes

# ======================
# LOAD MODELS
# ======================
feature_extractor = load_feature_extractor()

dqn = tf.keras.models.load_model(
    "../models/dqn_model.h5",
    compile=False
)

print("Models loaded ✔")

# ======================
# LOAD TEST DATA
# ======================
X = np.load("../extracted_features/features.npy").astype(np.float32)
y = np.load("../extracted_features/labels.npy")

num_samples = len(X)

correct = 0

# ======================
# TEST LOOP (ACCURACY)
# ======================
for i in range(num_samples):

    # ensure correct shape (1, 1280)
    state = X[i].reshape(1, -1)

    # DQN prediction
    q_values = dqn.predict(state, verbose=0)

    predicted_class = np.argmax(q_values)

    # check accuracy
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