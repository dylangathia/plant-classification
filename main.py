from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
multi_task_model = tf.keras.models.load_model("multi_task_plant_model.h5")

CROP_CLASSES = ['Bell Pepper', 'Potato']
DISEASE_CLASSES = ['Early Blight', 'Late Blight', 'Healthy', 'Bacterial Spot', 'Healthy']

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)

        crop_preds, disease_preds = multi_task_model.predict(img_batch)

        crop_class = CROP_CLASSES[np.argmax(crop_preds[0])]
        crop_confidence = float(np.max(crop_preds[0]))

        disease_class = DISEASE_CLASSES[np.argmax(disease_preds[0])]
        disease_confidence = float(np.max(disease_preds[0]))

        return {
            "crop": crop_class,
            "crop_confidence": crop_confidence,
            "disease": disease_class,
            "disease_confidence": disease_confidence,
        }
    except Exception as e:
        return {"error": str(e)}
