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
    page_icon="",
    layout="wide"
)

# ======================================
# Load Model
# ======================================

@st.cache_resource
def load_models():

    model_v2 = tf.keras.models.load_model(
        "models/MobileNetV2_FineTune.keras",
        compile=False
    )

    model_v3 = tf.keras.models.load_model(
        "models/MobileNetV3_FineTune.keras",
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
Upload gambar batik dan bandingkan hasil prediksi

"""
)

# ======================================
# Upload
# ======================================

uploaded_file = st.file_uploader(
    "Upload satu atau beberapa gambar batik ",
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
                class_names,
                model_type="v2"
            )

            pred_v3, conf_v3, prob_v3 = predict_image(
                model_v3,
                image,
                class_names,
                model_type="v3"
            )

        st.subheader("📊 Hasil Prediksi")

        hasil = pd.DataFrame({

            "Model": [
                "MobileNetV2",
                "MobileNetV3"
            ],

            "Prediksi": [
                pred_v2,
                pred_v3
            ],

            "Confidence": [
                f"{conf_v2*100:.2f}%",
                f"{conf_v3*100:.2f}%"
            ]

        })

        st.dataframe(
            hasil,
            use_container_width=True,
            hide_index=True
        )

        if conf_v2 > conf_v3:

            st.success(
                f"🏆 Confidence tertinggi diperoleh MobileNetV2 ({conf_v2*100:.2f}%)"
            )

        elif conf_v3 > conf_v2:

            st.success(
                f"🏆 Confidence tertinggi diperoleh MobileNetV3 ({conf_v3*100:.2f}%)"
            )

        else:

            st.info("Kedua model memiliki confidence yang sama.")

    st.divider()

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Top 5 Prediksi MobileNetV2")

        df_v2 = pd.DataFrame(

            top_predictions(
                prob_v2,
                class_names
            ),

            columns=[
                "Motif",
                "Confidence"
            ]

        )

        df_v2["Confidence"] = (
            df_v2["Confidence"] * 100
        ).round(2)

        st.dataframe(
            df_v2,
            use_container_width=True,
            hide_index=True
        )

    with col4:

        st.subheader("Top 5 Prediksi MobileNetV3")

        df_v3 = pd.DataFrame(

            top_predictions(
                prob_v3,
                class_names
            ),

            columns=[
                "Motif",
                "Confidence"
            ]

        )

        df_v3["Confidence"] = (
            df_v3["Confidence"] * 100
        ).round(2)

        st.dataframe(
            df_v3,
            use_container_width=True,
            hide_index=True
        )
