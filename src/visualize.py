import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

def plot_accuracy_and_loss(output_dir: str) -> None:
    """
    Plot training and validation accuracy/loss curves.

    Args:
        output_dir: Directory containing history.pkl and optionally history_fine.pkl.
    """
    # Load history
    with open(os.path.join(output_dir, 'history.pkl'), 'rb') as f:
        history = pickle.load(f)

    # Try to load fine-tuning history; fallback to empty if missing
    history_fine = {}
    try:
        with open(os.path.join(output_dir, 'history_fine.pkl'), 'rb') as f:
            history_fine = pickle.load(f)
    except FileNotFoundError:
        print("Warning: history_fine.pkl not found, using only history.pkl")

    # Combine histories
    combined_history = {
        'loss': history['loss'] + (history_fine.get('loss', [])),
        'accuracy': history['accuracy'] + (history_fine.get('accuracy', [])),
        'val_loss': history['val_loss'] + (history_fine.get('val_loss', [])),
        'val_accuracy': history['val_accuracy'] + (history_fine.get('val_accuracy', []))
    }

    # Accuracy curve
    plt.figure(figsize=(8, 6))
    plt.plot(combined_history['accuracy'], label='Training Accuracy')
    plt.plot(combined_history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(combined_history['loss'], label='Training Loss')
    plt.plot(combined_history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(output_dir: str) -> None:
    """
    Plot confusion matrix using test results.

    Args:
        output_dir: Directory containing test_results.pkl.
    """
    with open(os.path.join(output_dir, 'test_results.pkl'), 'rb') as f:
        results = pickle.load(f)
    
    y_true = results['y_true']
    y_pred_classes = results['y_pred_classes']
    class_names = results['class_names']

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(output_dir: str) -> None:
    """
    Plot precision-recall curves for each class.

    Args:
        output_dir: Directory containing test_results.pkl.
    """
    with open(os.path.join(output_dir, 'test_results.pkl'), 'rb') as f:
        results = pickle.load(f)
    
    y_true = results['y_true']
    y_pred_probs = results['y_pred_probs']
    class_names = results['class_names']

    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_pred_probs[:, i])
        plt.plot(recall, precision, label=f'{class_name} (AP={ap:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sample_images(data_dir: str, output_dir: str) -> None:
    """
    Plot sample images from each class with class name labels.

    Args:
        data_dir: Path to dataset directory (animal_data).
        output_dir: Directory containing class_names.pkl.
    """
    with open(os.path.join(output_dir, 'class_names.pkl'), 'rb') as f:
        class_names = pickle.load(f)

    num_samples = 5
    plt.figure(figsize=(num_samples * 3, len(class_names) * 3))
    
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) < num_samples:
            continue
        selected_images = images[:num_samples]
        
        for j, img_name in enumerate(selected_images):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).resize((224, 224))
            plt.subplot(len(class_names), num_samples, i * num_samples + j + 1)
            plt.imshow(img)
            plt.title(class_name, fontsize=10)  # Label each image with class name
            plt.axis('off')
    
    plt.suptitle('Sample Images from Each Class', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_images.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_visualizations(data_dir: str, output_dir: str) -> None:
    """
    Generate all required visualizations.

    Args:
        data_dir: Path to dataset directory.
        output_dir: Directory to save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Generating visualizations...")
    plot_accuracy_and_loss(output_dir)
    plot_confusion_matrix(output_dir)
    plot_precision_recall_curve(output_dir)
    plot_sample_images(data_dir, output_dir)
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    data_dir = "./animal_data"
    output_dir = "./outputs"
    generate_visualizations(data_dir, output_dir)