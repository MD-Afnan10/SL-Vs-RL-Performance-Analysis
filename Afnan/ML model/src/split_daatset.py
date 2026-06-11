import os
import random
import shutil

SOURCE_DIR = "dataset"
DEST_DIR = "dataset_split"

TRAIN_RATIO = 0.8

random.seed(42)

classes = os.listdir(SOURCE_DIR)

for class_name in classes:

    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    train_path = os.path.join(
        DEST_DIR,
        "train",
        class_name
    )

    test_path = os.path.join(
        DEST_DIR,
        "test",
        class_name
    )

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    images = os.listdir(class_path)

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    test_images = images[split_index:]

    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_path, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_path, img)
        )

print("Dataset split complete!")