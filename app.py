from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import joblib
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = load_model("prototype_model.keras")
label_encoder = joblib.load("label_encoder.pkl")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file type'}), 400

        pil_image = Image.open(file.stream).convert('RGB')
        pil_image = pil_image.resize((224, 224))
        image_array = preprocess_input(img_to_array(pil_image))
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions)

        predicted_artwork = label_encoder.inverse_transform([predicted_class_index])[0]

        return jsonify({'artwork': predicted_artwork})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()  
