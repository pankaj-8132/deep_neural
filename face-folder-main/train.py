import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath('src'))

from preprocess import load_data
from model import train_model, evaluate_model, visualize_predictions

# Path to the data directory
data_dir = 'data/train/'  

# Load the data (images and labels)
X, y = load_data(data_dir)

# Train the model and get the trained model and label encoder
model, le = train_model(X, y)

# Split data for evaluation
X_flat = X.reshape(X.shape[0], 64, 64, 1)
y_encoded = le.transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.3, random_state=42)

# Evaluate the model on the test set
evaluate_model(model, X_test, y_test)

# Visualize predictions
visualize_predictions(X_test, y_test, model, le)
