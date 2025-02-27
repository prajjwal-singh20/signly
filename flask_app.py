from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained Keras model (make sure the path is correct)
model = load_model('sign_language_model.keras')  # Change the path if needed

# Route for testing if the server is running
@app.route('/')
def home():
    return "Model is up and running!"

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Preprocess the image
        img = Image.open(file.stream)
        img = img.resize((28, 28))  # Resize to the model's expected input size (28x28)
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)  # Get the predicted class (letter/number)

        return jsonify({'predicted_class': int(predicted_class[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
