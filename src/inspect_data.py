import os
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def inspect_images(data_dir: str) -> None:
    """
    Visualize a few images from each class to check for quality or mislabeling.

    Args:
        data_dir: Path to animal_data with train/val/test subfolders.
    """
    class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant', 'Giraffe',
                   'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']
    num_samples = 3
    plt.figure(figsize=(num_samples * 3, len(class_names) * 3))
    
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, 'train', class_name)
        images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) < num_samples:
            continue
        selected_images = images[:num_samples]
        
        for j, img_name in enumerate(selected_images):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).resize((224, 224))
            plt.subplot(len(class_names), num_samples, i * num_samples + j + 1)
            plt.imshow(img)
            plt.title(f"{class_name} ({img_name})", fontsize=10)
            plt.axis('off')
    
    plt.suptitle('Sample Training Images', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('sample_training_images.png', dpi=300, bbox_inches='tight')
    plt.close()

def inspect_confusion_matrix(output_dir: str) -> None:
    """
    Plot confusion matrix to identify misclassified classes.

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
    plt.savefig('confusion_matrix_inspection.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    data_dir = "./animal_data"
    output_dir = "./outputs"
    inspect_images(data_dir)
    inspect_confusion_matrix(output_dir)
    print("Inspection plots saved: sample_training_images.png, confusion_matrix_inspection.png")