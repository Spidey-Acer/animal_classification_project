from model import build_model
from preprocess import load_and_preprocess_data
import tensorflow as tf

def train_model(data_dir, output_dir, epochs=20, batch_size=32):
    """
    Train the CNN model on the dataset.
    
    Args:
        data_dir (str): Path to dataset.
        output_dir (str): Directory to save model weights.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_and_preprocess_data(data_dir)
    num_classes = len(class_names)
    
    # Build and train model
    model = build_model(num_classes=num_classes)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)
    
    # Save model weights
    model.save_weights(os.path.join(output_dir, "model_weights.h5"))
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    return model, history, X_test, y_test, class_names

if __name__ == "__main__":
    import os
    data_dir = "../data/animal-data"
    output_dir = "../outputs"
    os.makedirs(output_dir, exist_ok=True)
    model, history, X_test, y_test, class_names = train_model(data_dir, output_dir)