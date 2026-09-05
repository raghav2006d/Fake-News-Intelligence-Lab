from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=20)
    source_url: Optional[str] = None
    explanation_method: str = Field(default="linear", pattern="^(linear|lime)$")


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    label: str
    confidence: float
    fake_probability: float
    real_probability: float
    stats: dict
    explanation: list
    model_name: str


class UrlRequest(BaseModel):
    url: str
