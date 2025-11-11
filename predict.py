import tensorflow as tf
import numpy as np
from PIL import Image

# ✅ Path to your model folder
MODEL_PATH = "models/model.savedmodel"

# ✅ Classes (update if needed)
CLASSES = ['Pizza', 'Cup Cakes', 'Samosa', 'Waffles', 'French Fries']

print("🔄 Loading model...")
model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
print("✅ Model loaded successfully (TFSMLayer)")

# ✅ Load and preprocess image
IMAGE_PATH = "test1.jpg"  # Change to your test image path
img = Image.open(IMAGE_PATH).convert("RGB").resize((224, 224))
img_array = np.array(img).astype(np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

print("🧠 Predicting...")
# ✅ Run prediction using correct endpoint and output key
pred_dict = model(img_array)
pred = pred_dict["sequential_3"].numpy()[0]

# ✅ Get predicted class
predicted_class = CLASSES[np.argmax(pred)]
confidence = np.max(pred) * 100

print(f"\n🍽️ Predicted Class: {predicted_class}")
print(f"📊 Confidence: {confidence:.2f}%")
