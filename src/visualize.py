import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from preprocess import get_sample_images
from typing import List, Tuple

def plot_sample_images(
    data_dir: str,
    class_names: List[str],
    output_dir: str,
    num_samples: int = 5
) -> None:
    """
    Plot sample images from each class.

    Args:
        data_dir: Path to dataset directory.
        class_names: List of class names.
        output_dir: Directory to save plot.
        num_samples: Number of samples per class.
    """
    samples = get_sample_images(data_dir, class_names, num_samples)
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(num_samples * 2, num_classes * 2))

    for i, (class_name, img) in enumerate(samples):
        row = i // num_samples
        col = i % num_samples
        ax = axes[row, col] if num_classes > 1 else axes[col]
        ax.imshow(img)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(class_name, fontsize=10, rotation=0, labelpad=40)
    
    plt.suptitle('Sample Images from Each Class', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_images.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    output_dir: str
) -> None:
    """
    Plot precision-recall curves for each class.

    Args:
        model: Trained Keras model.
        X_test: Test images.
        y_test: Test labels (one-hot encoded).
        class_names: List of class names.
        output_dir: Directory to save plot.
    """
    y_pred = model.predict(X_test, verbose=0)
    plt.figure(figsize=(10, 8))

    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
        ap = average_precision_score(y_test[:, i], y_pred[:, i])
        plt.plot(recall, precision, label=f'{class_name} (AP={ap:.2f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_additional_visualizations(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    data_dir: str,
    output_dir: str
) -> None:
    """
    Generate additional visualizations for the report.

    Args:
        model: Trained Keras model.
        X_test: Test images.
        y_test: Test labels.
        class_names: List of class names.
        data_dir: Path to dataset directory.
        output_dir: Directory to save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Generating additional visualizations...")
    plot_sample_images(data_dir, class_names, output_dir, num_samples=5)
    plot_precision_recall_curve(model, X_test, y_test, class_names, output_dir)
    print(f"Additional visualizations saved to {output_dir}")

if __name__ == "__main__":
    from train import train_model
    data_dir = "../animal_data"
    output_dir = "../outputs"
    model, history, X_test, y_test, class_names = train_model(data_dir, output_dir)
    generate_additional_visualizations(model, X_test, y_test, class_names, data_dir, output_dir)