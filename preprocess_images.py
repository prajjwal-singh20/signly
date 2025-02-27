import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Folder paths
image_folder = os.path.join(os.getcwd(), "images")
resized_image_folder = os.path.join(os.getcwd(), "resized_images")

# Ensure the resized images folder exists
if not os.path.exists(resized_image_folder):
    os.makedirs(resized_image_folder)
    print(f"Created folder: {resized_image_folder}")  # Debug print

# Get class names (folders only, ignoring .DS_Store and other files)
class_names = [f for f in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, f))]
class_names.sort()  # Ensure alphabetical order

print(f"Class Mapping: {dict(enumerate(class_names))}")  # Debugging output

# Save class names for Flask app to use later
np.save('class_names.npy', class_names)

# Initialize lists to hold image data and labels
images = []
labels = []

# Loop through each class folder (representing each gesture)
for class_name in class_names:
    class_folder = os.path.join(image_folder, class_name)

    # Create the corresponding folder in resized_image_folder
    resized_class_folder = os.path.join(resized_image_folder, class_name)
    os.makedirs(resized_class_folder, exist_ok=True)

    # Loop through each image in the folder
    for file_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, file_name)

        try:
            # Open and resize the image
            img = Image.open(img_path).resize((28, 28))  # Resize to 28x28
            img = img.convert("RGB")  # Ensure 3 color channels
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]

            # Save resized image
            resized_img_path = os.path.join(resized_class_folder, file_name)
            img.save(resized_img_path)

            # Add to dataset
            images.append(img_array)
            labels.append(class_names.index(class_name))  # Correct label mapping

        except Exception as e:
            print(f"Error processing image {file_name}: {e}")
            continue

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=len(class_names))
y_val = to_categorical(y_val, num_classes=len(class_names))
y_test = to_categorical(y_test, num_classes=len(class_names))

# Save the dataset
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)

print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}, Test data shape: {X_test.shape}")
print("Data saved successfully!")
