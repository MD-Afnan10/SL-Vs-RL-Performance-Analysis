import os
import shutil

source_dir = "./dataset-resized"
train_dir = "./dataset/train"
test_dir = "./dataset/test"

for class_name in os.listdir(source_dir):

    class_path = os.path.join(source_dir, class_name)
    images = os.listdir(class_path)

    split_index = int(len(images) * 0.8)

    train_images = images[:split_index]
    test_images = images[split_index:]

    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_dir, class_name, img)
        shutil.copy(src, dst)

    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_dir, class_name, img)
        shutil.copy(src, dst)

print("Done")