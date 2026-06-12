import os
import random

dataset_path = "./dataset-resized"

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):
        images = os.listdir(class_path)

        random.shuffle(images)

        for i, img in enumerate(images):
            old_path = os.path.join(class_path, img)
            new_name = f"{class_name}_{i+1}.jpg"
            new_path = os.path.join(class_path, new_name)

            os.rename(old_path, new_path)

print("done.")