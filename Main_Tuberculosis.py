!pip install gradio --quiet

import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ---------------------------------------------
# Load Model
# ---------------------------------------------
model_path = '/content/drive/My Drive/tb_ensemble_model.h5'
model = load_model(model_path, compile=False)

# Class labels
class_names = ['Normal', 'Tuberculosis']


# ---------------------------------------------
# Improved Grad-CAM Function
# ---------------------------------------------
def make_sharp_gradcam(model, img_array, conv_layer_name='block14_sepconv2_act'):

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


# ---------------------------------------------
# Prediction Function for Gradio
# ---------------------------------------------
def predict_tb(image):

    orig_shape = image.shape[:2]  # (height, width)

    img_resized = cv2.resize(image, (128, 128))
    img_input = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_input, axis=0)

    # Prediction
    prob = model.predict(img_array)[0][0]
    label = class_names[int(prob > 0.5)]
    confidence = prob if label == "Tuberculosis" else 1 - prob

    # Grad-CAM heatmap
    heatmap = make_sharp_gradcam(model, img_array)
    heatmap_resized = cv2.resize(heatmap, (orig_shape[1], orig_shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return label, round(confidence * 100, 2), overlay


# ---------------------------------------------
# Gradio Interface
# ---------------------------------------------
interface = gr.Interface(
    fn=predict_tb,
    inputs=gr.Image(type="numpy", label="Upload Chest X-Ray"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Number(label="Confidence (%)"),
        gr.Image(label="Grad-CAM Heatmap")
    ],
    title="Tuberculosis Detection from Chest X-Rays",
    description="Upload a chest X-ray image to detect Tuberculosis using an ensemble deep learning model with Grad-CAM explainability."
)

interface.launch(share=True)
