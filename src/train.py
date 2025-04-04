import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model
from preprocess import load_and_preprocess_data
from typing import Tuple

def train_model(
    data_dir: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 32
) -> Tuple:
    """
    Train the CNN model on the animal dataset with data augmentation.

    Args:
        data_dir (str): Path to dataset directory.
        output_dir (str): Directory to save model weights and outputs.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        Tuple containing:
            - model: Trained Keras model.
            - history: Training history.
            - X_test, y_test: Test data and labels.
            - class_names: List of class names.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_and_preprocess_data(data_dir)
    num_classes = len(class_names)

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    val_datagen = ImageDataGenerator()  # No augmentation for validation/test

    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    # Build and train model
    model = build_model(num_classes=num_classes)
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        steps_per_epoch=len(X_train) // batch_size,
        validation_steps=len(X_val) // batch_size,
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Save model weights
    model.save(os.path.join(output_dir, "model_weights.h5"))
    print(f"Model weights saved to {output_dir}/model_weights.h5")

    return model, history, X_test, y_test, class_names

if __name__ == "__main__":
    data_dir = "./animal_data"
    output_dir = "./outputs"
    model, history, X_test, y_test, class_names = train_model(data_dir, output_dir)