import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Folder paths
image_folder = r'A:\trials\sign\images'
resized_image_folder = r'A:\trials\sign\resized_images'

# Ensure the resized images folder exists
if not os.path.exists(resized_image_folder):
    os.makedirs(resized_image_folder)
    print(f"Created folder: {resized_image_folder}")  # Debug print

# Initialize lists to hold image data and labels
images = []
labels = []

# Get class names
class_names = os.listdir(image_folder)
class_names.sort()  # Ensure consistent order (e.g., A, B, C, ...)

# Debug: Print available class names
print(f"Classes found: {class_names}")

# Loop through each class folder (representing each gesture)
for label, class_name in enumerate(class_names):
    class_folder = os.path.join(image_folder, class_name)
    
    # Create the corresponding folder in resized_image_folder
    resized_class_folder = os.path.join(resized_image_folder, class_name)
    
    # Ensure the class folder exists, create any necessary parent directories
    if not os.path.exists(resized_class_folder):
        try:
            os.makedirs(resized_class_folder)  # Create the class folder
            print(f"Created folder: {resized_class_folder}")  # Debug print
        except Exception as e:
            print(f"Error creating folder {resized_class_folder}: {e}")
            continue
    
    # Loop through each image in the folder
    for file_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, file_name)
        
        try:
            # Open and resize the image
            img = Image.open(img_path).resize((28, 28))  # Resize to 28x28
            print(f"Resized image: {file_name}")  # Debug print

            # Save the resized image
            resized_img_path = os.path.join(resized_class_folder, file_name)
            img.save(resized_img_path)
            print(f"Saved image {file_name} to {resized_class_folder}")  # Debug print
        except Exception as e:
            print(f"Error processing image {file_name}: {e}")
            continue
        
        # Convert image to numpy array and normalize
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        
        # Add the image and label to the lists
        images.append(img_array)
        labels.append(label)

# Print the number of images and labels to verify
print(f"Total images loaded: {len(images)}")
print(f"Total labels: {len(labels)}")

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# If there are no images, raise an error
if images.shape[0] == 0:
    raise ValueError("No images were loaded. Check if the image paths are correct.")

# Split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# One-hot encode the labels (convert labels to categorical format)
y_train = to_categorical(y_train, num_classes=len(class_names))
y_val = to_categorical(y_val, num_classes=len(class_names))
y_test = to_categorical(y_test, num_classes=len(class_names))

print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}, Test data shape: {X_test.shape}")

# Save the data to .npy files
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)

np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)

print("Data saved successfully!")
