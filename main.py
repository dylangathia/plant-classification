from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"] to be strict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("potatoes.h5")
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@app.get("/")
async def ping():
    return {"message": "Hello I'm alive!"}

def read_file_as_image(data) -> np.ndarray:
    """
    Read file as image
    """
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):
    image = read_file_as_image(await file.read()) #if 100 callers call this endpoint, 100 files will be read
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    np.max(predictions[0])

    confidence = np.max(predictions[0])

    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
    }
   

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

#TF Serving IS YET TO BE IMPLEMENTED