from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from typing import Tuple

def build_model(input_shape: Tuple[int, int, int] = (224, 224, 3), num_classes: int = 15) -> models.Model:
    """
    Build ResNet50-based model for animal classification.

    Args:
        input_shape: Input image shape (height, width, channels).
        num_classes: Number of classes (default: 15).

    Returns:
        Compiled Keras model.
    """
    # Load pre-trained ResNet50
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Initially freeze all layers (train.py will handle fine-tuning)
    base_model.trainable = False

    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Model Architecture:")
    model.summary()

    return model

if __name__ == "__main__":
    model = build_model(num_classes=15)