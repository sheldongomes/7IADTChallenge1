from keras.models import load_model
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from pathlib import Path

# Configuration and paths
BASE_DIR = Path(__file__).parent.parent
TEST_DIR = BASE_DIR / "data" / "test"
MODEL_PATH = BASE_DIR / "models" / "cancer_vision_model.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Directory Validation
print(f"Searching in: {TEST_DIR}")
if not TEST_DIR.exists():
    raise FileNotFoundError(f"Folder not found: {TEST_DIR}")
print("Folders:", [p.name for p in TEST_DIR.iterdir() if p.is_dir()])

# Data Generator (Only rescale)
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"Classes: {test_gen.class_indices}")
print(f"Testing: {test_gen.samples} images\n")

# Loading model
model = load_model(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}")

# Recompiling
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall']
)

# Evaluating
print("EVALUATING TEST DATA...\n")
results = model.evaluate(test_gen, verbose=1)

# Displaying results
if len(results) == 2:
    loss, acc = results
    print(f"Loss: {loss:.4f} | Accuracy: {acc:.4f}")
elif len(results) == 4:
    loss, acc, precision, recall = results
    print(f"Loss: {loss:.4f} | Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
else:
    print("Results:", results)