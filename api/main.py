from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup

from api.database import init_db, log_prediction, recent_predictions
from api.model_service import FakeNewsModelService
from api.schemas import BatchPredictionRequest, PredictionRequest, UrlRequest
from src.cloud_storage import cloud_status, upload_artifacts


app = FastAPI(
    title="Fake News Detection API",
    description="ML API for article classification, batch scoring, and article URL extraction.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = FakeNewsModelService()


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": model_service.model_name}


@app.get("/model-info")
def model_info() -> dict:
    return {"model": model_service.model_name, "metrics": model_service.metrics}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    prediction = model_service.predict_one(
        payload.text,
        explanation_method=payload.explanation_method,
    )
    log_prediction(payload.text, prediction, source_url=payload.source_url)
    return prediction


@app.post("/batch-predict")
def batch_predict(payload: BatchPredictionRequest) -> dict:
    results = model_service.predict_batch(payload.texts)
    for text, prediction in zip(payload.texts, results):
        log_prediction(text, prediction)
    return {"results": results}


@app.get("/prediction-history")
def prediction_history(limit: int = 25) -> dict:
    return {"results": recent_predictions(limit)}


@app.get("/cloud/status")
def get_cloud_status() -> dict:
    return cloud_status()


@app.post("/cloud/sync-artifacts")
def sync_cloud_artifacts() -> dict:
    return {"uploaded": upload_artifacts()}


@app.post("/extract-url")
def extract_url(payload: UrlRequest) -> dict:
    try:
        response = requests.get(payload.url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {exc}") from exc

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.extract()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(paragraphs)
    return {"title": title, "text": text[:20000]}
