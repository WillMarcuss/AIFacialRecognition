import os
import cv2
import numpy as np
import pickle

# Constants
DATA_DIR = "data/"
IMG_SIZE = 48
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_data(data_dir):
    X, y = [], []
    for label, emotion in enumerate(EMOTIONS):
        folder_path = os.path.join(data_dir, emotion)
        print(f"Loading {emotion} images...")
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(label)
            except Exception as e:
                print(f"Error with image {img_path}: {e}")
    return np.array(X), np.array(y)

# Load and preprocess the data
X_train, y_train = load_data(os.path.join(DATA_DIR, 'train'))
X_test, y_test = load_data(os.path.join(DATA_DIR, 'test'))

# Normalize images
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape for CNN input
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Save processed data
with open("data_preprocessed.pkl", "wb") as f:
    pickle.dump((X_train, y_train, X_test, y_test), f)
print("Data preprocessed and saved!")
