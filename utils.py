import json
import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as preprocess_v2
)

from tensorflow.keras.applications.mobilenet_v3 import (
    preprocess_input as preprocess_v3
)

# ==========================================
# Load Nama Kelas
# ==========================================

def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================
# Preprocessing
# ==========================================

def preprocess_image(
    image,
    model_type="v2",
    target_size=(224, 224)
):

    image = image.convert("RGB")
    image = image.resize(target_size)

    img = np.array(image).astype(np.float32)

    img = np.expand_dims(img, axis=0)

    if model_type.lower() == "v2":
        img = preprocess_v2(img)
    else:
        img = preprocess_v3(img)

    return img


# ==========================================
# Prediksi
# ==========================================

def predict_image(
    model,
    image,
    class_names,
    model_type="v2"
):

    img = preprocess_image(
        image,
        model_type=model_type
    )

    prediction = model.predict(
        img,
        verbose=0
    )

    prediction = prediction[0]

    predicted_index = np.argmax(prediction)

    confidence = float(prediction[predicted_index])

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence, prediction


# ==========================================
# Top-K Prediction
# ==========================================

def top_predictions(
    prediction,
    class_names,
    top_k=5
):

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
