import tensorflow as tf
import numpy as np
import pickle
import os

def evaluate_model(data_dir: str, output_dir: str) -> None:
    """
    Evaluate the trained model on the test set.

    Args:
        data_dir: Path to dataset directory (animal_data).
        output_dir: Directory containing model and class names.
    """
    # Load class names
    with open(os.path.join(output_dir, 'class_names.pkl'), 'rb') as f:
        class_names = pickle.load(f)

    # Load model
    model = tf.keras.models.load_model(os.path.join(output_dir, 'model.keras'))

    # Test data generator
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input
    )
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        classes=class_names,
        shuffle=False
    )

    # Evaluate (limit to ~120 samples)
    test_generator.reset()
    steps = int(np.ceil(120 / test_generator.batch_size))
    test_loss, test_accuracy = model.evaluate(test_generator, steps=steps)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Predictions for confusion matrix and PR curves
    test_generator.reset()
    y_pred = model.predict(test_generator, steps=steps)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes[:120]  # Limit to 120 samples
    y_pred_probs = y_pred

    # Save results
    with open(os.path.join(output_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump({
            'y_true': y_true,
            'y_pred_classes': y_pred_classes,
            'y_pred_probs': y_pred_probs,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'class_names': class_names
        }, f)

if __name__ == "__main__":
    data_dir = "./animal_data"
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    evaluate_model(data_dir, output_dir)