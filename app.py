import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd

from utils import (
    load_class_names,
    predict_image,
    top_predictions
)

# ======================================
# Konfigurasi Halaman
# ======================================

st.set_page_config(
    page_title="Klasifikasi Motif Batik",
    page_icon="🎨",
    layout="wide"
)

# ======================================
# Load Model
# ======================================

@st.cache_resource
def load_models():

    model_v2 = tf.keras.models.load_model(
        "models/MobileNetV2_FineTune.h5",
        compile=False
    )

    model_v3 = tf.keras.models.load_model(
        "models/MobileNetV3_FineTune.h5",
        compile=False
    )

    return model_v2, model_v3


@st.cache_data
def load_classes():

    return load_class_names(
        "models/class_names.json"
    )


model_v2, model_v3 = load_models()
class_names = load_classes()

# ======================================
# Judul
# ======================================

st.title("Klasifikasi Motif Batik Indonesia")

st.markdown(
"""
Aplikasi ini melakukan klasifikasi motif batik menggunakan dua model
deep learning yaitu **MobileNetV2** dan **MobileNetV3**.
"""
)

# ======================================
# Upload Gambar
# ======================================

uploaded_file = st.file_uploader(
    "Upload gambar batik",
    type=["jpg", "jpeg", "png"]
)

# ======================================
# Prediksi
# ======================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(
            image,
            caption="Gambar yang diupload",
            use_container_width=True
        )

    with col2:

        with st.spinner("Melakukan prediksi..."):

            pred_v2, conf_v2, prob_v2 = predict_image(
                model_v2,
                image,
                class_names
            )

            pred_v3, conf_v3, prob_v3 = predict_image(
                model_v3,
                image,
                class_names
            )

        st.subheader("📊 Hasil Prediksi")

        hasil = pd.DataFrame({

            "Model":[
                "MobileNetV2",
                "MobileNetV3"
            ],

            "Prediksi":[
                pred_v2,
                pred_v3
            ],

            "Confidence":[
                f"{conf_v2*100:.2f}%",
                f"{conf_v3*100:.2f}%"
            ]

        })

        st.dataframe(
            hasil,
            use_container_width=True,
            hide_index=True
        )

        # Model terbaik

        if conf_v2 > conf_v3:

            st.success(
                f"🏆 Model terbaik: MobileNetV2 ({conf_v2*100:.2f}%)"
            )

        elif conf_v3 > conf_v2:

            st.success(
                f"🏆 Model terbaik: MobileNetV3 ({conf_v3*100:.2f}%)"
            )

        else:

            st.info("Kedua model memiliki confidence yang sama.")

    # ======================================
    # Top 5 Prediction
    # ======================================

    st.divider()

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Top 5 MobileNetV2")

        top5 = top_predictions(
            prob_v2,
            class_names
        )

        df = pd.DataFrame(
            top5,
            columns=[
                "Motif",
                "Confidence"
            ]
        )

        df["Confidence"] = (
            df["Confidence"]*100
        ).round(2)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    with col4:

        st.subheader("Top 5 MobileNetV3")

        top5 = top_predictions(
            prob_v3,
            class_names
        )

        df = pd.DataFrame(
            top5,
            columns=[
                "Motif",
                "Confidence"
            ]
        )

        df["Confidence"] = (
            df["Confidence"]*100
        ).round(2)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )