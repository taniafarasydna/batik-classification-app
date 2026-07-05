import json
import numpy as np
import tensorflow as tf
from PIL import Image

# ==============================
# Load Nama Kelas
# ==============================

def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================
# Preprocessing
# ==============================

def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)

    img = np.array(image).astype("float32")

    img = np.expand_dims(img, axis=0)

    # Sesuai preprocessing saat training
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    return img


# ==============================
# Prediksi
# ==============================

def predict_image(model, image, class_names):

    img = preprocess_image(image)

    prediction = model.predict(img, verbose=0)

    confidence = float(np.max(prediction))

    predicted_index = int(np.argmax(prediction))

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence, prediction[0]


# ==============================
# Top K Prediction
# ==============================

def top_predictions(prediction, class_names, top_k=5):

    idx = np.argsort(prediction)[::-1][:top_k]

    results = []

    for i in idx:

        results.append(
            (
                class_names[i],
                float(prediction[i])
            )
        )

    return results