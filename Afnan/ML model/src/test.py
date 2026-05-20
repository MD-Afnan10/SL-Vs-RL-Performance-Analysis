import numpy as np
import tensorflow as tf
from utils import load_feature_extractor, preprocess_image, classes

# ======================
# LOAD MODELS (SAFE)
# ======================

feature_extractor = load_feature_extractor()

dqn = tf.keras.models.load_model(
    "../models/dqn_model.h5",
    compile=False   
)

# ======================
# IMAGE INPUT
# ======================

img_path = r"F:\ML\ML model\test_images\test.jpg"

img = preprocess_image(img_path)

# ensure batch dimension
img = np.expand_dims(img, axis=0)

# ======================
# FEATURE EXTRACTION
# ======================

features = feature_extractor.predict(img, verbose=0)

# ensure correct shape (1280,)
features = np.array(features).reshape(1, -1)

# ======================
# DQN PREDICTION
# ======================

q_values = dqn.predict(features, verbose=0)

predicted_class = np.argmax(q_values)

# ======================
# OUTPUT
# ======================

print("Predicted class:", classes[predicted_class])