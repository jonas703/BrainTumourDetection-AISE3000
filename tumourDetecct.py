import torch
import numpy as np
import PIL
from PIL import Image
import os
import cv2

def load_images_from_folder(folder_path, image_size=(224, 224)):
    """
    Load images from the specified folder and its subfolders,
    resize them to the specified size, and convert to pixel arrays
    """
    images = []
    labels = []
    
    # Get all subfolders (classes)
    class_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    for class_name in class_folders:
        class_path = os.path.join(folder_path, class_name)
        print(f"Loading images from class: {class_name}")
        
        # Get all images in the class folder
        for image_name in os.listdir(class_path):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_path, image_name)
                
                # Read and preprocess the image
                image = cv2.imread(image_path)
                if image is not None:
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Resize image
                    image = cv2.resize(image, image_size)
                    # Normalize pixel values to [0, 1]
                    image = image / 255.0
                    
                    images.append(image)
                    labels.append(class_name)
    
    return np.array(images), np.array(labels)

# Load training data
train_path = "BrainTumourDetection-AISE3000/train"
X_train, y_train = load_images_from_folder(train_path)
print(f"Training data shape: {X_train.shape}")
print(f"Number of training labels: {len(y_train)}")

# Load testing data
test_path = "BrainTumourDetection-AISE3000/test"
X_test, y_test = load_images_from_folder(test_path)
print(f"Testing data shape: {X_test.shape}")
print(f"Number of testing labels: {len(y_test)}")

