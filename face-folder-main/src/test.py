import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from preprocess import load_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('trained_model.h5')

# Load test data
data_dir = 'data/test/'  # Path to your test data directory
X, y = load_data(data_dir)  # Load data from the test folder

# Reshape and normalize test data as done during training
X_flat = X.reshape(X.shape[0], 64, 64, 1)  # Reshape for grayscale images

# Encode the labels (same as during training)
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Fit the encoder to labels

# Split data for evaluation (optional, if you want to see individual performance)
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.3, random_state=42)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Print the testing accuracy and loss
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Predict the labels for the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the predicted class labels

# Classification report (precision, recall, F1-score)
print("\nClassification Report:")
# Convert class labels to strings to avoid TypeError
target_names = [str(cls) for cls in le.classes_]
print(classification_report(y_test, y_pred_classes, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Visualize predictions vs actual labels for the first few test samples
plt.figure(figsize=(12, 12))
for i in range(9):  # Display the first 9 test samples
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
    plt.title(f'True: {le.inverse_transform([y_test[i]])[0]}, Pred: {le.inverse_transform([y_pred_classes[i]])[0]}')
    plt.axis('off')
plt.show()
