import numpy as np
import tensorflow as tf
import joblib

from utils import (
    load_feature_extractor,
    preprocess_image,
    classes
)

# ======================
# LOAD MODELS
# ======================

feature_extractor = load_feature_extractor()

dqn = tf.keras.models.load_model(
    "Models/dqn_model.h5",
    compile=False
)

scaler = joblib.load(
    "Models/scaler.pkl"
)

print("Models loaded ✔")

# ======================
# IMAGE INPUT
# ======================

img_path = r"test_images\test.jpg"

# ======================
# PREPROCESS IMAGE
# ======================

img = preprocess_image(img_path)

# Add batch dimension
img = np.expand_dims(
    img,
    axis=0
)

# ======================
# FEATURE EXTRACTION
# ======================

features = feature_extractor.predict(
    img,
    verbose=0
)

# Convert to shape (1, 1280)
features = features.reshape(
    1,
    -1
)

# ======================
# FEATURE NORMALIZATION
# ======================

features = scaler.transform(
    features
)

# ======================
# DQN PREDICTION
# ======================

q_values = dqn.predict(
    features,
    verbose=0
)

predicted_class = np.argmax(
    q_values
)

confidence = np.max(
    q_values
)

# ======================
# OUTPUT
# ======================

print("\n======================")
print("PREDICTION RESULT")
print("======================")

print("Q Values:")
print(q_values)

print(
    "\nPredicted Class:",
    classes[predicted_class]
)

print(
    "Predicted Label:",
    predicted_class
)

print(
    "Confidence Score:",
    round(float(confidence), 4)
)