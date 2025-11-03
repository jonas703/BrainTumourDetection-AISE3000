import torch
import numpy as np
import PIL
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt




def load_images_from_folder(folder_path, image_size=(224, 224), validation_split=0):
   
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
                    
    images = np.array(images)
    labels = np.array(labels)
    
    if validation_split > 0:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, 
            test_size=validation_split,
            random_state=42,
            stratify=labels  # Ensure balanced split across classes
        )
        return X_train, X_val, y_train, y_val
    else:
        return images, labels





# Visualization code

def visualize_samples(images, labels, num_samples_per_class=5):
    """
    Visualize sample images from each class in a grid
    """
    # Get unique classes
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_classes, num_samples_per_class, 
                            figsize=(15, 3*num_classes))
    fig.suptitle('Sample Images from Each Class', fontsize=16)
    
    # Plot samples from each class
    for i, class_name in enumerate(unique_classes):
        # Get indices for current class
        class_indices = np.where(labels == class_name)[0]
        
        # Randomly select samples
        selected_indices = np.random.choice(class_indices, 
                                          min(num_samples_per_class, len(class_indices)), 
                                          replace=False)
        
        # Plot each sample
        for j, idx in enumerate(selected_indices):
            axes[i, j].imshow(images[idx])
            axes[i, j].axis('off')
            if j == 0:  # Only show class name on first image of each row
                axes[i, j].set_title(class_name)
    
    plt.tight_layout()
    plt.show()


 # Load training data
train_path = "BrainTumourDetection-AISE3000/train"
X_train, y_train= load_images_from_folder(train_path)
print(f"Training data shape: {X_train.shape}")
print(f"Number of training labels: {len(y_train)}")

#Data is initially split into 80% train and 20% test so splitting the test set gives an 80/10/10 train/val/test split
# Load testing data and split into test and validation sets
test_path = "BrainTumourDetection-AISE3000/test"
X_test, X_val, y_test, y_val = load_images_from_folder(test_path, validation_split=0.5)
print(f"Testing data shape: {X_test.shape}")
print(f"Number of testing labels: {len(y_test)}")   
print(f"Validation data shape: {X_val.shape}")
print(f"Number of validation labels: {len(y_val)}")

visualize = True

if visualize == True:
    # Visualize Data Samples
    print("\nVisualizing training samples:")
    visualize_samples(X_train, y_train)
    print("\nVisualizing validation samples:")
    visualize_samples(X_val, y_val)
    print("\nVisualizing testing samples:")
    visualize_samples(X_test, y_test)

