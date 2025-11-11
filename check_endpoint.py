import tensorflow as tf
from tensorflow.python.framework import convert_to_constants

# Load the model
model = tf.keras.layers.TFSMLayer("models/model.savedmodel", call_endpoint="serving_default")

# Print the available signatures (endpoints)
loaded = tf.saved_model.load("models/model.savedmodel")

print("\n=== Available Signatures ===")
for key, value in loaded.signatures.items():
    print(f"Endpoint: {key}")
    print(f"Details: {value}")
