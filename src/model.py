from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3), num_classes=5):
    """
    Build a Convolutional Neural Network for image classification.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels).
        num_classes (int): Number of classes in the dataset.
    
    Returns:
        model: Compiled Keras model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # Prevent overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

if __name__ == "__main__":
    model = build_model(num_classes=5)  # Adjust num_classes based on your dataset