import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Any

def plot_accuracy_and_loss(history: Any, output_dir: str) -> None:
    """
    Plot training and validation accuracy and loss curves.

    Args:
        history: Keras training history object.
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    output_dir: str
) -> None:
    """
    Plot confusion matrix for test set predictions.

    Args:
        model: Trained Keras model.
        X_test: Test images.
        y_test: Test labels (one-hot encoded).
        class_names: List of class names.
        output_dir: Directory to save plot.
    """
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(
    model: Any,
    history: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    output_dir: str
) -> None:
    """
    Evaluate the model and generate core visualizations.

    Args:
        model: Trained Keras model.
        history: Training history object.
        X_test: Test images.
        y_test: Test labels.
        class_names: List of class names.
        output_dir: Directory to save visualizations.
    """
    print("Generating core visualizations...")
    plot_accuracy_and_loss(history, output_dir)
    plot_confusion_matrix(model, X_test, y_test, class_names, output_dir)
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    from train import train_model
    data_dir = "./animal_data"
    output_dir = "./outputs"
    model, history, X_test, y_test, class_names = train_model(data_dir, output_dir)
    evaluate_model(model, history, X_test, y_test, class_names, output_dir)