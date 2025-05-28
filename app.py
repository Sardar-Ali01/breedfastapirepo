# import uvicorn
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# import numpy as np
# import tensorflow as tf
# import scipy.stats
# from tensorflow.keras.preprocessing import image
# import io
# from PIL import Image
# from zipfile import ZipFile

# MODEL_PATH = "resnet101_best_model.keras"
# DATASET_ZIP = "cleaned_pets_dataset_blur_filtered.zip"

# # Load class names from zip file once
# with ZipFile(DATASET_ZIP, 'r') as zip_ref:
#     all_files = zip_ref.namelist()
#     class_names = sorted(set(f.split('/')[0] for f in all_files if f.endswith('/') and f.count('/') == 1))

# # Load model once
# model = tf.keras.models.load_model(MODEL_PATH)

# app = FastAPI()

# # Adjust this to your frontend URL, or allow all for testing
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Your React app URL here
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# def preprocess_image_bytes(image_bytes, target_size=(224, 224)):
#     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     img = img.resize(target_size)
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# @app.post("/predict")
# async def predict_breed_api(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     img_array = preprocess_image_bytes(image_bytes)

#     logits = model.predict(img_array)[0]
#     temperature = 0.2
#     scaled_logits = logits / temperature
#     probs = tf.nn.softmax(scaled_logits).numpy()

#     confidence = float(np.max(probs))
#     entropy = float(scipy.stats.entropy(probs))
#     class_idx = int(np.argmax(probs))

#     threshold = 0.92
#     entropy_threshold = 2.5

#     if confidence < threshold or entropy > entropy_threshold:
#         predicted_breed = "Unknown breed"
#     else:
#         predicted_breed = class_names[class_idx]
        

#     return {
#         "breed": predicted_breed,
#         "confidence": confidence,
#         "entropy": entropy,
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import scipy.stats
from PIL import Image
import io

MODEL_PATH = "resnet101_best_model.keras"

# Define class names explicitly in the correct order
class_names = [
    'Abyssinian', 'American Robin', 'Bald Eagle', 'Bengal', 'Birman',
    'Blue JayBombay', 'Common Raven', 'american_bulldog',
    'american_pit_bull_terrier', 'basset_hound', 'beagle'
]

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI()

# CORS middleware to allow requests from your frontend on Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://petfolioo.netlify.app"],  # Frontend URL from Netlify
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image_bytes(image_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict_breed_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = preprocess_image_bytes(image_bytes)

    logits = model.predict(img_array)[0]
    temperature = 0.2
    scaled_logits = logits / temperature
    probs = tf.nn.softmax(scaled_logits).numpy()

    confidence = float(np.max(probs))
    entropy = float(scipy.stats.entropy(probs))
    class_idx = int(np.argmax(probs))

    threshold = 0.92
    entropy_threshold = 2.5

    if confidence < threshold or entropy > entropy_threshold:
        predicted_breed = "Unknown breed"
    else:
        predicted_breed = class_names[class_idx]

    return {
        "breed": predicted_breed,
        "confidence": confidence,
        "entropy": entropy,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
