from fastapi import FastAPI, File, UploadFile, Request
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi import Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends
import json

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
multi_task_model = tf.keras.models.load_model("C:/potato-disease/models/multi_task_plant_model.h5")

# Class names
CROP_CLASSES = ['Bell Pepper', 'Potato']
DISEASE_CLASSES = ['Early Blight', 'Late Blight', 'Healthy', 'Bacterial Spot', 'Healthy']


with open("C:/potato-disease/api/recommendations.json") as f:
    RECOMMENDATIONS = json.load(f)

app.mount("/static", StaticFiles(directory="C:/potato-disease/api/static/"), name="static")

templates = Jinja2Templates(directory="C:/potato-disease/api/templates")

# Route: Home page
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    user = request.cookies.get("user")  # simulate session (you can expand later)
    return templates.TemplateResponse("index.html", {"request": request, "user": user})


def read_file_as_image(data) -> np.ndarray:

    """Reads a file and converts it to an numpy array."""

    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize the image to [0, 1] range
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

        recommendation = RECOMMENDATIONS.get(disease_class, "No recommendations available")

        return {
            "crop": crop_class,
            "crop_confidence": crop_confidence,
            "disease": disease_class,
            "disease_confidence": disease_confidence,
            "recommendation": recommendation
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
