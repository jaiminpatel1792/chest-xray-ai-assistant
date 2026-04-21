from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from app.backend.schemas import PredictionResponse
from app.services.inference_service import InferenceService
from app.services.report_service import build_report
from app.services.gradcam_service import generate_gradcam_base64
from src.data.dataset import DEFAULT_LABELS


app = FastAPI(title="CheXpert X-ray Inference API")

inference_service = InferenceService()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    result = inference_service.predict(image)

    report = build_report(
        labels=DEFAULT_LABELS,
        probs=result["raw_probs"],
        ground_truth=None,
        threshold=0.5,
    )

    gradcam_base64 = generate_gradcam_base64(
        image=image,
        model=inference_service.model,
        probs=result["raw_probs"],
    )

    return PredictionResponse(
        top_label=result["top_label"],
        top_probability=result["top_probability"],
        probabilities=result["probabilities"],
        report=report,
        image_filename=file.filename,
        gradcam_base64=gradcam_base64,
    )