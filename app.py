import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd

# ===== App Config =====
IMG_SIZE = 224
MODEL_PATH = "dr_model.h5"
LAST_CONV_LAYER = "top_conv"
DR_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# ===== Custom CSS for Styling =====
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    h1, h2, h3, h4 {
        color: #1a237e;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ===== Header =====
st.markdown("<h1 style='text-align: center;'>👁️ Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Upload a retinal image to predict DR stage and visualize attention via Grad-CAM</h4>", unsafe_allow_html=True)
st.markdown("---")

# ===== Load Model =====
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ===== Image Preprocessing =====
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# ===== Grad-CAM Function =====
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    base_model = model.layers[0]
    inputs = tf.keras.Input(shape=(224, 224, 3))
    conv_output = base_model.get_layer(last_conv_layer_name).output
    base_output = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(base_output)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
    grad_model = tf.keras.Model(inputs=base_model.input, outputs=[conv_output, outputs])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy(), preds.numpy()

# ===== Upload Section =====
st.markdown("### 📤 Upload a Retinal Image")
uploaded_file = st.file_uploader("Choose a PNG or JPG file", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = preprocess_image(image)
    heatmap, preds = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)

    # ===== Prediction Section =====
    pred_label = np.argmax(preds)
    confidence = preds[0][pred_label]

    st.markdown("---")
    st.markdown("### 🧠 Prediction Result")
    st.markdown(f"<h2 style='text-align: center; color: #2e7d32;'>{DR_LABELS[pred_label]}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: gray;'>Confidence: {confidence*100:.2f}%</h4>", unsafe_allow_html=True)

    st.markdown("### 🔍 Class Probabilities")
    probs_df = pd.DataFrame(preds, columns=DR_LABELS)
    st.bar_chart(probs_df.T)

    # ===== Visual Output Section =====
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📷 Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### 🔥 Grad-CAM Heatmap")
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        original = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
        superimposed = cv2.addWeighted(original, 0.7, heatmap_color, 0.3, 0)
        st.image(superimposed, use_container_width=True)

else:
    st.info("Please upload a retinal image to begin.", icon="👆")
