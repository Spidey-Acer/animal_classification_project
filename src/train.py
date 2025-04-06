import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import pickle
import numpy as np

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 15
EPOCHS = 30  # Reduced for initial training; fine-tuning adds more
DATA_DIR = './animal_data'
OUTPUT_DIR = './outputs'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class names
class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant', 'Giraffe', 
               'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']
with open(os.path.join(OUTPUT_DIR, 'class_names.pkl'), 'wb') as f:
    pickle.dump(class_names, f)

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

# Data generators for training, validation, and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.3
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.3
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    subset='training',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    subset='validation',
    shuffle=False
)
test_generator = test_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    shuffle=False
)

# Build model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze initially

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
print("Initial Model Architecture:")
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Initial training
print("Starting initial training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Save initial history
with open(os.path.join(OUTPUT_DIR, 'history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

# Fine-tuning
print("Starting fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:100]:  # Freeze first 100 layers
    layer.trainable = False

# Recompile with lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print fine-tuning summary
print("Fine-Tuning Model Architecture:")
model.summary()

# Continue training
history_fine = model.fit(
    train_generator,
    epochs=EPOCHS,
    initial_epoch=len(history.epoch),
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Save fine-tuning history
with open(os.path.join(OUTPUT_DIR, 'history_fine.pkl'), 'wb') as f:
    pickle.dump(history_fine.history, f)

# Evaluate on test set
test_generator.reset()
steps = int(np.ceil(119 / BATCH_SIZE))  # Match preprocess.py's 119 test samples
test_loss, test_accuracy = model.evaluate(test_generator, steps=steps)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Save model
model.save(os.path.join(OUTPUT_DIR, 'model.keras'))