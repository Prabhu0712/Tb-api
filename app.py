from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import gdown

app = Flask(__name__)
CORS(app)

# ✅ Download model from Google Drive if not found locally
model_path = 'tb_ensemble_model.h5'
if not os.path.exists(model_path):
    print("Model not found locally. Downloading from Google Drive...")
    file_id = '17s1KTYDWmWK8uOBzm68EAXh27CruIDOb'  # 🔁 Replace this with your actual file ID
    gdown.download(f'https://drive.google.com/file/d/17s1KTYDWmWK8uOBzm68EAXh27CruIDOb/view?usp=drive_link', model_path, quiet=False)

# ✅ Load the model
model = load_model(model_path, compile=False)

# ✅ Class labels
class_names = ['Normal', 'Tuberculosis']

# ✅ Grad-CAM function
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

# ✅ TB prediction logic
def predict_tb(image):
    orig_shape = image.shape[:2]
    img_resized = cv2.resize(image, (128, 128))
    img_input = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_input, axis=0)
    prob = model.predict(img_array)[0][0]
    label = class_names[int(prob > 0.5)]
    confidence = prob if label == "Tuberculosis" else 1 - prob
    heatmap = make_sharp_gradcam(model, img_array, conv_layer_name='block14_sepconv2_act')
    heatmap_resized = cv2.resize(heatmap, (orig_shape[1], orig_shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0.6, heatmap_color, 0.4, 0)
    return label, round(confidence * 100, 2), overlay

# ✅ API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    image_np = np.array(image)
    label, confidence, heatmap = predict_tb(image_np)
    _, buffer = cv2.imencode('.png', heatmap)
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({
        "label": label,
        "confidence": confidence,
        "heatmap": heatmap_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
