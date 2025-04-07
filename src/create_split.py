import os
import shutil
import random

def create_split(data_dir: str, output_dir: str) -> None:
    """
    Create train/val/test subfolders matching preprocess.py's 840/241/119 split.

    Args:
        data_dir: Path to animal_data with class subfolders.
        output_dir: Path to create train/val/test subfolders.
    """
    random.seed(42)  # For reproducibility
    class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant', 'Giraffe',
                   'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']
    images_per_class = 80
    train_per_class = 56  # ~70% of 80
    val_per_class = 16   # ~20% of 80
    test_per_class = 8   # ~10% of 80

    # Create directories
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for cls in class_names:
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

    # Copy images
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) < images_per_class:
            print(f"Warning: {cls} has only {len(images)} images, expected {images_per_class}")
        random.shuffle(images)
        
        # Assign images
        train_images = images[:train_per_class]
        val_images = images[train_per_class:train_per_class + val_per_class]
        test_images = images[train_per_class + val_per_class:train_per_class + val_per_class + test_per_class]
        
        # Copy to respective folders
        for img in train_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(output_dir, 'train', cls, img))
        for img in val_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(output_dir, 'val', cls, img))
        for img in test_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(output_dir, 'test', cls, img))
    
    # Verify counts
    for split in ['train', 'val', 'test']:
        count = sum(len(os.listdir(os.path.join(output_dir, split, cls))) for cls in class_names)
        print(f"{split.capitalize()}: {count} images")

if __name__ == "__main__":
    data_dir = "./animal_data"
    output_dir = "./animal_data"
    create_split(data_dir, output_dir)