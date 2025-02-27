from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


# Load the model from the saved .keras file
model = load_model('sign_language_model.keras')

# Print the model summary (optional, just to verify it loaded correctly)
model.summary()

# Example: Use the model to make predictions (using new data)
# Assuming you have some new data `X_new_data`
# predictions = model.predict(X_new_data)

# You can replace the above line with real input data, like image data
