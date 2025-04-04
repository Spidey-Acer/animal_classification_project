import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def plot_accuracy(history, output_dir):
    """Plot training and validation accuracy."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))
    plt.close()

def plot_confusion_matrix(model, X_test, y_test, class_names, output_dir):
    """Plot confusion matrix for test set predictions."""
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

if __name__ == "__main__":
    from train import train_model
    data_dir = "../data/animal-data"
    output_dir = "../outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    model, history, X_test, y_test, class_names = train_model(data_dir, output_dir)
    plot_accuracy(history, output_dir)
    plot_confusion_matrix(model, X_test, y_test, class_names, output_dir)