# Fake News Intelligence Lab

Fake News Intelligence Lab is an end-to-end data science, machine learning, and cloud-integrated web application for detecting misleading news articles. The project converts an experimental notebook into a production-style ML system with a React frontend, FastAPI inference service, explainable predictions, model training workflows, database logging, Docker, CI/CD, and a free AWS-compatible cloud layer using LocalStack.

## Highlights

- Trained fake news classifier using TF-IDF features and Logistic Regression
- Fine-tuning pipeline for DistilBERT article classification
- Explainable AI using linear TF-IDF coefficient explanations and optional LIME
- React + Vite frontend with a modern premium dashboard interface
- FastAPI backend with prediction, batch prediction, URL extraction, history, and cloud artifact endpoints
- PostgreSQL prediction logging with SQLite fallback for local development
- Free AWS-compatible S3 artifact storage using LocalStack, so no AWS account or billing is required
- DVC pipeline definition for reproducible model workflows
- MLflow experiment tracking support
- Docker Compose stack for API, frontend, PostgreSQL, and LocalStack
- GitHub Actions for backend tests, frontend build, Docker build, and deploy workflows
- Deployment configs for Render, Railway, Vercel, Netlify, AWS EC2, and AWS ECS

## Problem Statement

Fake news detection is a binary text classification task. Given a news title/article body, the system predicts whether the content is likely fake or real. The application also explains which words or phrases influenced the model, stores prediction activity, and exposes cloud-style model artifact storage.

## Dataset

The project uses the Fake and Real News dataset containing `Fake.csv` and `True.csv`.

Local dataset archive:

```text
fake news dataset ml.zip
```

Labels:

```text
0 = Fake News
1 = Real News
```

## Implemented Production Model

The deployed production model is a classical NLP pipeline optimized for fast API inference:

```text
Raw article text
-> text cleaning
-> TF-IDF vectorization
-> Logistic Regression classifier
-> probability score
-> explanation terms
```

Techniques used:

- lowercasing
- URL, hashtag, mention, number, and symbol removal
- whitespace normalization
- TF-IDF feature extraction
- unigram and bigram features
- stop-word removal
- stratified train/test split
- class-balanced Logistic Regression
- MLflow metric logging
- model serialization with Joblib

Current trained baseline metrics:

```text
Accuracy:  0.9888
Precision: 0.9843
Recall:    0.9923
F1 Score:  0.9883
ROC-AUC:   0.9992
Samples:   44,898
```

This is the model currently loaded by the FastAPI service from:

```text
models/fake_news_pipeline.joblib
```

## Notebook Research and Ensemble Modeling

The original notebook explored a broader fake news detection workflow before the project was converted into a clean production app. That research workflow included EDA, NLP feature engineering, multi-model comparison, ensemble modeling, and a real-time prediction prototype.

Notebook techniques included:

- spaCy-based text cleaning and lemmatization
- TF-IDF and CountVectorizer text representations
- unigram and bigram feature extraction
- text length, word count, sentence count, and average word length
- part-of-speech counts for nouns, verbs, adjectives, and adverbs
- named entity count
- punctuation features such as exclamation and question mark counts
- TextBlob sentiment polarity and subjectivity
- feature scaling with `StandardScaler`
- sparse feature stacking with TF-IDF plus engineered numeric features
- stratified train, validation, and test splitting
- cross-validation using F1 scoring
- ROC-AUC, precision, recall, F1-score, and confusion matrix evaluation

Models explored in the notebook:

```text
Logistic Regression
Random Forest
Gradient Boosting
Support Vector Machine
Multinomial Naive Bayes
MLP Neural Network
Soft Voting Ensemble
DistilBERT transformer demo / fine-tuning path
```

The ensemble workflow used:

```text
TF-IDF text features
+ text_length
+ word_count
+ sentiment polarity
+ sentiment subjectivity
-> scaling for numeric features
-> feature stacking
-> individual model training
-> Gradient Boosting validation with early stopping
-> soft VotingClassifier ensemble
-> 3-fold cross-validation
-> comprehensive test evaluation
```

The production API currently serves the lighter TF-IDF + Logistic Regression model because it is fast, easy to deploy, explainable, and reliable for a portfolio web service. The notebook ensemble and DistilBERT training scripts remain documented as advanced experimentation paths that can be promoted to production later.

## DistilBERT Pipeline

The project includes a transformer training script for fine-tuning DistilBERT:

```text
src/train_distilbert.py
```

It uses Hugging Face Transformers, Hugging Face Datasets, PyTorch, dynamic padding, stratified splits, accuracy/F1 evaluation, and MLflow reporting.

Run:

```bash
pip install -r requirements-transformer.txt
python src/train_distilbert.py --sample-size 8000 --epochs 2 --batch-size 8
```

Quick smoke run:

```bash
python src/train_distilbert.py --sample-size 500 --epochs 1 --batch-size 4 --max-length 128
```

## Explainable AI

The API supports two explanation modes:

```text
linear
lime
```

Linear explanation extracts active TF-IDF terms, multiplies each TF-IDF value by the Logistic Regression coefficient, returns the top weighted words/phrases, and marks whether each term pushed toward fake or real.

LIME explanation uses local perturbations around the input text to estimate model-agnostic term importance.

## Backend API

Endpoints:

```text
GET  /health
GET  /model-info
POST /predict
POST /batch-predict
POST /extract-url
GET  /prediction-history
GET  /cloud/status
POST /cloud/sync-artifacts
```

Core backend files:

```text
api/main.py
api/model_service.py
api/database.py
api/schemas.py
src/text_processing.py
src/explain.py
src/cloud_storage.py
```

## Frontend

The frontend is built with React, Vite, CSS, Lucide icons, and Recharts.

Features:

- article text input
- URL article extraction
- text/CSV upload
- fake/real prediction card
- confidence visualization
- model metrics dashboard
- explanation method selector
- signal terms display
- recent prediction history
- free AWS-compatible cloud status panel

## Free AWS-Compatible Cloud Layer

This project avoids AWS billing by using LocalStack. LocalStack provides AWS-compatible services locally through Docker.

Implemented cloud-style workflow:

```text
Model artifacts + metrics
-> API cloud sync endpoint
-> LocalStack S3 bucket
-> AWS-compatible object storage workflow
```

No AWS account is required for this local cloud workflow. The same code can later point to real AWS S3 by removing `AWS_ENDPOINT_URL` and using real AWS credentials.

Cloud files:

```text
src/cloud_storage.py
scripts/sync_artifacts_to_s3.py
docker-compose.yml
```

## Database Logging

Prediction logs are stored through SQLAlchemy.

Local development:

```text
SQLite fallback: prediction_logs.db
```

Docker/cloud:

```text
PostgreSQL
```

Logged fields include timestamp, label, confidence, fake probability, real probability, model name, source URL, text preview, and word count.

## MLOps Workflow

DVC pipeline:

```text
dvc.yaml
```

Stages:

```text
train
train_distilbert
```

MLflow logs baseline model metrics and supports experiment comparison between classical ML and transformer runs.

Run DVC locally:

```bash
pip install -r requirements-mlops.txt
dvc repro
```

## Architecture

```text
React Frontend
  |
  | HTTP JSON
  v
FastAPI Backend
  |
  |-- TF-IDF + Logistic Regression model
  |-- LIME / linear explanation service
  |-- URL extraction service
  |-- SQLAlchemy logging layer
  |
  |-- PostgreSQL / SQLite
  |
  |-- LocalStack S3 artifact storage
```

Docker services:

```text
frontend
api
postgres
localstack
```

## Local Setup

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Train baseline model:

```bash
python src/train.py
```

Run API:

```bash
uvicorn api.main:app --reload
```

Run frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

API docs:

```text
http://localhost:8000/docs
```

## Docker Setup

Run the full local cloud stack:

```bash
docker compose up --build
```

Services:

```text
Frontend:   http://localhost:5173
API:        http://localhost:8000
API Docs:   http://localhost:8000/docs
PostgreSQL: localhost:5432
LocalStack: localhost:4566
```

Sync model artifacts to LocalStack S3:

```bash
python scripts/sync_artifacts_to_s3.py
```

## Deployment

Free/no-card local cloud:

```text
Docker Compose + PostgreSQL + LocalStack S3
```

Free hosted options:

```text
Frontend: Vercel or Netlify
Backend: Render or Railway free tier, depending on current provider limits
Database: hosted PostgreSQL free tier, depending on current provider limits
```

AWS-ready options:

```text
AWS EC2
AWS ECS Fargate
AWS S3
AWS RDS PostgreSQL
```

Real AWS may create charges, so this repo defaults to LocalStack for free AWS-compatible development.

Deployment files:

```text
render.yaml
railway.json
frontend/vercel.json
frontend/netlify.toml
deployment/aws-ec2.md
deployment/aws-ecs.md
deployment/render.md
deployment/railway.md
deployment/vercel.md
deployment/netlify.md
```

## CI/CD

GitHub Actions:

```text
.github/workflows/ci.yml
.github/workflows/deploy-render-vercel.yml
.github/workflows/deploy-aws-ecs.yml
```

CI validates backend dependencies, unit tests, API import, frontend install, frontend production build, Docker image build, and the DVC pipeline graph.

## Testing

Run tests:

```bash
pytest tests
```

On this Windows environment, pytest required a `PYTHONPATH` override because a stray global `py.py` file interfered with pytest imports:

```powershell
$env:PYTHONPATH='C:\Users\raghav\AppData\Local\Programs\Python\Python312\Lib\site-packages'
python -m pytest tests
```

## Project Structure

```text
api/                 FastAPI backend and database logging
src/                 ML, preprocessing, explanation, cloud storage code
frontend/            React + Vite frontend
models/              Trained baseline model bundle
reports/             Training metrics
scripts/             Artifact sync utilities
deployment/          Cloud deployment guides
tests/               Unit tests
.github/workflows/   CI/CD workflows
```

## Resume Summary

Built an end-to-end fake news detection platform using Python, scikit-learn, FastAPI, React, PostgreSQL, Docker, MLflow, DVC, and LocalStack. Trained a TF-IDF + Logistic Regression model achieving 98.8% accuracy and 0.999 ROC-AUC on 44k+ news articles, added explainable AI with linear coefficient analysis and LIME, implemented REST APIs for real-time and batch inference, logged prediction history in PostgreSQL, and created an AWS-compatible no-cost cloud workflow using LocalStack S3 for model artifact storage. Added CI/CD workflows and deployment configs for Render, Railway, Vercel, Netlify, AWS EC2, and AWS ECS.
