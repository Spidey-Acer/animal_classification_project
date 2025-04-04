import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(data_dir, img_size=(224, 224)):
    """
    Load images from the dataset and preprocess them.
    
    Args:
        data_dir (str): Path to the dataset directory.
        img_size (tuple): Target image size (height, width).
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: Preprocessed data splits.
        class_names: List of class names.
    """
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize and normalize
                img = cv2.resize(img, img_size)
                img = img.astype('float32') / 255.0  # Normalize to [0, 1]
                images.append(img)
                labels.append(label)
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # One-hot encode labels
    y = to_categorical(y, num_classes=len(class_names))
    
    # Split dataset: 70% train, 20% validation, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")
    print(f"Number of classes: {len(class_names)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_names

if __name__ == "__main__":
    data_dir = "../data/animal_data"
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_and_preprocess_data(data_dir)