import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications import ResNet50
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration and paths
BASE_DIR = Path(__file__).parent.parent
TRAIN_DIR = BASE_DIR / "data" / "train"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "cancer_vision_model.h5"

EPOCHS = 10
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Validating folder data
print(f"Search for images in: {TRAIN_DIR}")
if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Folder not found: {TRAIN_DIR}")
print("Folders found:", [p.name for p in TRAIN_DIR.iterdir() if p.is_dir()])

# Data augmentation (this technique will increase the size of training dataset by creating modified copies of exist data)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print(f"\Classes: {train_gen.class_indices}")
print(f"Training: {train_gen.samples} images")
print(f"Validation: {val_gen.samples} images\n")

# Modeling
base = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False

model = Sequential([
    base,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall']
)

# Training
print("STARTING TRAINING...\n")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1
)

# Saving model
model.save(MODEL_PATH)
print(f"\nMODEL SAVED: {MODEL_PATH}")

# Plot of the training
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Training', color='#1f77b4', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation', color='#ff7f0e', linewidth=2)
    ax1.set_title('Loss during training', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history.history['accuracy'], label='Training', color='#2ca02c', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validation', color='#d62728', linewidth=2)
    ax2.set_title('Accuracy during training', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = MODEL_DIR / "training_history.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"GRAPHIC SAVED: {plot_path}")
    plt.show()

plot_training_history(history)

print("\nTRAINING COMPLETED WITH SUCCESS!")