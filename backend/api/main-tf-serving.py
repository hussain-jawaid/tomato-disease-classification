import asyncio
import json
import os
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

app = FastAPI()

TF_SERVING_URL = os.getenv(
    "TF_SERVING_URL",
    "http://localhost:8501/v1/models/tomato:predict",
)
IMAGE_SIZE = (256, 256)
CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]


@app.get("/ping")
async def ping():
    return {"message": "pong"}


def read_file_as_image(data: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(data)) as image:
            image = image.convert("RGB").resize(
                IMAGE_SIZE,
                Image.Resampling.BILINEAR,
            )
            return np.asarray(image, dtype=np.float32)
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image",
        ) from exc


def predict_with_tf_serving(image: np.ndarray) -> np.ndarray:
    payload = {
        "signature_name": "serve",
        "instances": [image.tolist()],
    }
    request = Request(
        TF_SERVING_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read())
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"TensorFlow Serving returned HTTP {exc.code}: {error_body}",
        ) from exc
    except URLError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to TensorFlow Serving: {exc.reason}",
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="TensorFlow Serving returned invalid JSON",
        ) from exc

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=502,
            detail="TensorFlow Serving returned an unexpected response",
        )

    output = result.get("outputs", result.get("predictions"))
    if isinstance(output, dict):
        if len(output) != 1:
            raise HTTPException(
                status_code=502,
                detail="TensorFlow Serving returned an unexpected number of outputs",
            )
        output = next(iter(output.values()))

    try:
        scores = np.asarray(output, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="TensorFlow Serving returned invalid prediction scores",
        ) from exc

    if scores.shape == (len(CLASS_NAMES),):
        scores = scores[np.newaxis, :]
    if scores.shape != (1, len(CLASS_NAMES)):
        raise HTTPException(
            status_code=502,
            detail="TensorFlow Serving returned scores with an unexpected shape",
        )

    return scores[0]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    scores = await asyncio.to_thread(predict_with_tf_serving, image)

    predicted_class = CLASS_NAMES[int(np.argmax(scores))]
    confidence = float(np.max(scores) * 100)
    return {"predicted_class": predicted_class, "confidence": confidence}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
