from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# ✅ Load model once
MODEL_PATH = "models/model.savedmodel"
model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

# ✅ Class names (same order as training)
CLASSES = ['Pizza', 'Cup Cakes', 'Samosa', 'Waffles', 'French Fries']

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No file uploaded")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction="No selected file")

    # Save uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess image
    img = Image.open(filepath).convert("RGB").resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred_dict = model(img_array)
    pred = pred_dict["sequential_3"].numpy()[0]
    predicted_class = CLASSES[np.argmax(pred)]
    confidence = np.max(pred) * 100

    result = f"{predicted_class} ({confidence:.2f}%)"

    # ✅ Pass relative path (not full one)
    image_filename = os.path.basename(filepath)
    return render_template('index.html', prediction=result, image_path=image_filename)


if __name__ == '__main__':
    app.run(debug=True)
