import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import urllib.request
from PIL import Image
from io import BytesIO
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "cancer_vision_model.h5"
IMG_URL = [{
    "diagnostic": "MALIGNANT",
    "url": "https://prod-images-static.radiopaedia.org/images/628131/5ed61b5d938cd7adac6a6e635689f2_big_gallery.jpg",
    "description": ""
},
{
    "diagnostic": "BENIGN",
    "url": "https://prod-images-static.radiopaedia.org/images/634594/612563f060a68d7899335e483e8f09_big_gallery.jpg",
    "description": "Fibroadenoma that can be a false positive"
},
{
    "diagnostic": "BENIGN",
    "url": "https://prod-images-static.radiopaedia.org/images/7226779/4b7862667bed0856d1054a4263b11b_big_gallery.jpg",
    "description": "This is a Mammography and can confuse the model"
},
{
    "diagnostic": "BENIGN",
    "url": "https://prod-images-static.radiopaedia.org/images/45173184/7328bd5c74ce2f14a5801eadf643c3_big_gallery.jpeg",
    "description": ""
},
{
    "diagnostic": "BENIGN",
    "url": "https://prod-images-static.radiopaedia.org/images/52552974/4c1c504353bce912c66cfc76c62190.JPG",
    "description": ""
}]


# Loading model
model = tf.keras.models.load_model(MODEL_PATH)

# Download and load image
for entry in range(len(IMG_URL)):

    response = urllib.request.urlopen(IMG_URL[entry]['url'])
    img = Image.open(BytesIO(response.read()))
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    pred = model.predict(x)[0][0]
    label = "MALIGNANT" if pred > 0.5 else "BENIGN"
    prob = pred if pred > 0.5 else 1 - pred

    print(f"Diagnostic: {label}")
    print(f"Reliability: {prob:.2%}")