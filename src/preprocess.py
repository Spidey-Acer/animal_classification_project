import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from typing import Tuple, List
import random

class DatasetError(Exception):
    """Custom exception for dataset-related errors."""
    pass

def load_and_preprocess_data(data_dir: str, img_size: Tuple[int, int] = (224, 224), max_images_per_class: int = 80) -> Tuple[np.ndarray, ...]:
    """
    Load and preprocess images from the animal dataset, limiting to max_images_per_class per class.

    Args:
        data_dir (str): Path to the dataset directory (e.g., '../animal_data').
        img_size (tuple): Target image size (height, width).
        max_images_per_class (int): Maximum number of images per class.

    Returns:
        Tuple containing:
            - X_train, X_val, X_test: Preprocessed image arrays.
            - y_train, y_val, y_test: One-hot encoded labels.
            - class_names: List of class names.

    Raises:
        DatasetError: If the dataset is invalid or images cannot be loaded.
    """
    if not os.path.exists(data_dir):
        raise DatasetError(f"Dataset directory {data_dir} does not exist.")

    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    if len(class_names) < 5:
        raise DatasetError(f"Expected at least 5 classes, found {len(class_names)}.")

    print(f"Found {len(class_names)} classes: {class_names}")

    # Load images, limiting to max_images_per_class per class
    class_counts = {}
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(img_files)  # Shuffle to avoid bias
        img_files = img_files[:max_images_per_class]  # Limit to 80 images
        class_counts[class_name] = len(img_files)

        for img_name in img_files:
            img_path = os.path.join(class_dir, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Failed to load {img_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = cv2.resize(img, img_size)
                img = img.astype('float32') / 255.0  # Normalize to [0, 1]
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Warning: Error processing {img_path}: {e}")

    if not images:
        raise DatasetError("No valid images found in the dataset.")

    print("Class distribution after limiting:")
    for class_name, count in class_counts.items():
        print(f" - {class_name}: {count} images")

    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    y = to_categorical(y, num_classes=len(class_names))

    # Split dataset: 70% train, 20% validation, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.33, stratify=y_temp, random_state=42
    )

    print(f"Dataset loaded successfully:")
    print(f" - Training: {X_train.shape[0]} samples")
    print(f" - Validation: {X_val.shape[0]} samples")
    print(f" - Test: {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test, class_names

def get_sample_images(data_dir: str, class_names: List[str], num_samples: int = 5) -> List[Tuple[str, np.ndarray]]:
    """
    Collect sample images for visualization.

    Args:
        data_dir (str): Path to dataset directory.
        class_names (List[str]): List of class names.
        num_samples (int): Number of sample images per class.

    Returns:
        List of tuples (class_name, image_array).
    """
    samples = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(img_files)
        for img_name in img_files[:num_samples]:
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                samples.append((class_name, img))
    return samples

if __name__ == "__main__":
    data_dir = "./animal_data"
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_and_preprocess_data(data_dir)
        print(f"Classes: {class_names}")
    except DatasetError as e:
        print(f"Error: {e}")