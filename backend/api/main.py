import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from pathlib import Path

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_models" / "1" / "model.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)
class_names = json.loads(
    Path(__file__).resolve().with_name("class_names.json").read_text(encoding="utf-8")
)


@app.get("/ping")
async def ping():
    return {"message": "pong"}


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))
    return np.array(image)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(image_batch)

    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])*100
    return {"predicted_class": predicted_class, "confidence": float(confidence)}
    pass


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
