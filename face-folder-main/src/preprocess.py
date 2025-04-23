import os
import cv2
import numpy as np

def load_data(data_dir):
    categories = ['fake', 'real']  # Adjust categories as necessary
    data = []
    labels = []

    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            img = cv2.resize(img, (64, 64))  # Resize to 64x64
            data.append(img)
            labels.append(label)

    X = np.array(data).astype('float32') / 255.0  # Normalize pixel values
    y = np.array(labels)

    return X, y
