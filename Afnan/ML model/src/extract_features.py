import os
import sys
import numpy as np
from utils import load_feature_extractor, preprocess_image, get_label_from_folder

model = load_feature_extractor()

dataset_path = "F:\ML\ML model\dataset"
features = []
labels = []

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)

    for file in os.listdir(class_path):
        img_path = os.path.join(class_path, file)
        img = preprocess_image(img_path)

        img = np.expand_dims(img, axis=0)
        feature = model.predict(img)[0]   # shape (1280,)

        features.append(feature)
        labels.append(get_label_from_folder(class_name))

features = np.array(features)
labels = np.array(labels)

os.makedirs("../extracted_features", exist_ok=True)

np.save("../extracted_features/features.npy", features)
np.save("../extracted_features/labels.npy", labels)

print("Feature extraction complete!")
print("Feature shape:", features.shape)