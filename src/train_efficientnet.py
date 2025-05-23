import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import pickle
import numpy as np
import tensorflow_addons as tfa

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 15
EPOCHS_INITIAL = 20
EPOCHS_FINE = 80
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
    brightness_range=[0.7, 1.3],
    channel_shift_range=20.0,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

# Data generators
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    shuffle=False
)

# Build model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze initially

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
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
early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

# Initial training
print("Starting initial training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS_INITIAL,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

# Save initial history
with open(os.path.join(OUTPUT_DIR, 'history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

# Fine-tuning with cyclical learning rate
print("Starting fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:20]:  # Unfreeze more layers
    layer.trainable = False

# Cyclical learning rate
steps_per_epoch = len(train_generator)
clr = tfa.optimizers.CyclicalLearningRate(
    initial_learning_rate=1e-5,
    maximal_learning_rate=2e-5,
    step_size=2 * steps_per_epoch,
    scale_fn=lambda x: 1.0,
    scale_mode="cycle"
)

# Recompile with cyclical learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=clr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print fine-tuning summary
print("Fine-Tuning Model Architecture:")
model.summary()

# Continue training
history_fine = model.fit(
    train_generator,
    epochs=EPOCHS_INITIAL + EPOCHS_FINE,
    initial_epoch=len(history.epoch),
    validation_data=val_generator,
    callbacks=[early_stopping]
)

# Save fine-tuning history
with open(os.path.join(OUTPUT_DIR, 'history_fine.pkl'), 'wb') as f:
    pickle.dump(history_fine.history, f)

# Evaluate on test set
test_generator.reset()
steps = int(np.ceil(120 / BATCH_SIZE))
test_loss, test_accuracy = model.evaluate(test_generator, steps=steps)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Save model
model.save(os.path.join(OUTPUT_DIR, 'model.keras'))