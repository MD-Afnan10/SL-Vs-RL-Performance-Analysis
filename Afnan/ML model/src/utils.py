import tensorflow as tf
import numpy as np
import cv2

IMG_SIZE = (224, 224)
classes = ["cardboard", "plastic", "paper", "metal", "trash", "glass"]

def load_feature_extractor():
    model = tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet",
        pooling='avg'   #gives 1280 vector
    )
    return model

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0 #normalize the pixel values(0-1)
    return img

def get_label_from_folder(folder_name):
    return classes.index(folder_name)