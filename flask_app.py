from flask import Flask, request, jsonify, send_from_directory, render_template, url_for
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import time

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the trained model
model_path = 'sign_language_model.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Ensure it's in the project directory.")

model = load_model(model_path)

# Load class names from saved file
class_names_path = 'class_names.npy'
if not os.path.exists(class_names_path):
    raise FileNotFoundError(f"Class names file '{class_names_path}' not found. Run preprocessing first.")

class_names = np.load(class_names_path, allow_pickle=True)

# Serve the main frontend page
@app.route('/')
def home():
    return render_template('index.html')

# Serve Static Files (CSS, JS, Images)
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory("static", path)

# Route for making predictions
@app.route('/predict', methods=['POST'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        start_time = time.time()  # Track time

        # Preprocess the image
        img = Image.open(file.stream)
        img = img.resize((28, 28))
        img = img.convert("RGB")
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make Predictions
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]

        # Convert index to class label
        predicted_class = class_names[predicted_index]

        end_time = time.time()  # End tracking time
        print(f"🕒 Prediction Time: {round(end_time - start_time, 3)} sec")

        return jsonify({'predicted_class': str(predicted_class)})

    except Exception as e:
        return jsonify({'error': str(e)})
# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
