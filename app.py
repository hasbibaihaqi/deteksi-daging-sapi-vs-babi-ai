import os
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── Load model ────────────────────────────────────────
MODEL_PATH = 'models/model_sapi_babi.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded.")
except Exception as e:
    print(f"⚠️  Model not found: {e}")
    model = None

IMG_SIZE = 299   # Xception input size

def preprocess_image(image_path):
    """
    Preprocess gambar agar persis sama dengan standar model:
    - Resize ke 299x299
    - Menggunakan preprocess_input dari Xception (skala [-1, 1])
    """
    from tensorflow.keras.applications.xception import preprocess_input
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    
    # Menggunakan preprocess_input bawaan Xception yang lebih akurat daripada sekadar / 255.0
    arr = preprocess_input(arr)
    
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Simpan dengan nama unik agar tidak konflik
    ext      = os.path.splitext(secure_filename(file.filename))[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if model is None:
        return render_template('result.html',
                               image_url=filepath,
                               result="Model Belum Dilatih",
                               result_en="untrained",
                               confidence="0.00",
                               meat_type="unknown")

    arr  = preprocess_image(filepath)
    preds = model.predict(arr, verbose=0)[0]
    
    # Handle both binary (1 output) and categorical (2 outputs) models
    if len(preds) > 1:
        # Categorical: [prob_babi, prob_sapi] based on folder names (Babi=0, Sapi=1)
        babi_prob = float(preds[0])
        sapi_prob = float(preds[1])
        if sapi_prob > babi_prob:
            result_label = "Daging Sapi"
            result_en    = "beef"
            confidence   = sapi_prob * 100
            meat_type    = "sapi"
        else:
            result_label = "Daging Babi"
            result_en    = "pork"
            confidence   = babi_prob * 100
            meat_type    = "babi"
    else:
        # Binary: prob is probability of class 1 (Sapi)
        prob = float(preds[0])
        if prob > 0.5:
            result_label = "Daging Sapi"
            result_en    = "beef"
            confidence   = prob * 100
            meat_type    = "sapi"
        else:
            result_label = "Daging Babi"
            result_en    = "pork"
            confidence   = (1 - prob) * 100
            meat_type    = "babi"

    return render_template('result.html',
                           image_url=filepath,
                           result=result_label,
                           result_en=result_en,
                           confidence=f"{confidence:.1f}",
                           meat_type=meat_type)


if __name__ == '__main__':
    app.run(debug=True)