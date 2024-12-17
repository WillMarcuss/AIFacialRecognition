import pickle
import tensorflow as tf

# Load preprocessed data
with open("data_preprocessed.pkl", "rb") as f:
    _, _, X_test, y_test = pickle.load(f)

# Load the trained model
model = tf.keras.models.load_model("emotion_model.h5")

# Convert labels to one-hot encoding
y_test = tf.keras.utils.to_categorical(y_test, num_classes=7)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
