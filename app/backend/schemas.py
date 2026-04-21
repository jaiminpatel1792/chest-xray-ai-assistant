from typing import Dict, Optional

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    top_label: str
    top_probability: float
    probabilities: Dict[str, float]
    report: str
    image_filename: Optional[str] = None
    gradcam_base64: Optional[str] = None