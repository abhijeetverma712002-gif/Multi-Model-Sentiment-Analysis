"""
REST API for Sentiment Analysis Models
FastAPI-based API endpoint for production use
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pickle
import xgboost as xgb
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

app = FastAPI(
    title="Multi-Model Sentiment Analysis API",
    description="Production-ready API for sentiment analysis with 4 trained models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configurations
MODEL_CONFIGS = {
    "3-class": {
        "model_file": "xgboost_sentiment_model.json",
        "vectorizer_file": "tfidf_vectorizer.pkl",
        "mappings_file": "label_mappings.pkl",
        "classes": ["negative", "neutral", "positive"]
    },
    "binary": {
        "model_file": "binary_sentiment_model.json",
        "vectorizer_file": "binary_vectorizer.pkl",
        "mappings_file": "binary_mappings.pkl",
        "classes": ["negative", "positive"]
    },
    "5-star": {
        "model_file": "fiveStar_model.json",
        "vectorizer_file": "fiveStar_vectorizer.pkl",
        "mappings_file": "fiveStar_mappings.pkl",
        "classes": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    },
    "emotion": {
        "model_file": "emotion_bert_model",
        "mappings_file": "emotion_mappings.pkl",
        "classes": ["anger", "disgust", "fear", "joy", "sadness", "surprise"],
        "model_type": "bert"
    }
}

# Load models cache
loaded_models = {}

def load_model_api(model_name: str):
    """Load and cache models"""
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    if not os.path.exists(config["model_file"]):
        raise HTTPException(status_code=404, detail=f"Model files not found for '{model_name}'")
    
    try:
        if config.get("model_type") == "bert":
            model = BertForSequenceClassification.from_pretrained(config["model_file"])
            tokenizer = BertTokenizer.from_pretrained(config["model_file"])
            with open(config["mappings_file"], 'rb') as f:
                mappings = pickle.load(f)
            loaded_models[model_name] = (model, tokenizer, mappings['id2label'])
        else:
            model = xgb.XGBClassifier()
            model.load_model(config["model_file"])
            with open(config["vectorizer_file"], 'rb') as f:
                vectorizer = pickle.load(f)
            with open(config["mappings_file"], 'rb') as f:
                mappings = pickle.load(f)
            loaded_models[model_name] = (model, vectorizer, mappings['id2label'])
        
        return loaded_models[model_name]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def predict_api(text: str, model_name: str) -> Dict:
    """Make prediction"""
    model, vectorizer, id2label = load_model_api(model_name)
    config = MODEL_CONFIGS[model_name]
    
    if config.get("model_type") == "bert":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs = vectorizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            pred_proba = probs[0].cpu().numpy()
    else:
        text_tfidf = vectorizer.transform([text])
        pred_id = model.predict(text_tfidf)[0]
        pred_proba = model.predict_proba(text_tfidf)[0]
    
    probabilities = {config["classes"][i]: float(pred_proba[i]) for i in range(len(config["classes"]))}
    
    return {
        'text': text,
        'model': model_name,
        'prediction': id2label[pred_id],
        'confidence': float(pred_proba[pred_id]),
        'probabilities': probabilities
    }

# Request/Response models
class PredictionRequest(BaseModel):
    text: str
    model: str = "3-class"

class BatchPredictionRequest(BaseModel):
    texts: List[str]
    model: str = "3-class"

class PredictionResponse(BaseModel):
    text: str
    model: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

# API Endpoints
@app.get("/")
async def root():
    """API root"""
    return {
        "message": "Multi-Model Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/models": "List available models",
            "/predict": "Single text prediction",
            "/batch": "Batch predictions",
            "/health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": len(loaded_models)}

@app.get("/models")
async def list_models():
    """List available models"""
    models_info = []
    for name, config in MODEL_CONFIGS.items():
        exists = os.path.exists(config["model_file"])
        models_info.append({
            "name": name,
            "classes": config["classes"],
            "available": exists
        })
    return {"models": models_info}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single text prediction"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        result = predict_api(request.text, request.model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Batch predictions"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if len(request.texts) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 texts per batch")
    
    try:
        results = []
        for text in request.texts:
            if text.strip():
                result = predict_api(text, request.model)
                results.append(result)
        
        return {
            "model": request.model,
            "total": len(results),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
async def compare_models(text: str):
    """Compare prediction across all models"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        results = {}
        for model_name in MODEL_CONFIGS.keys():
            try:
                result = predict_api(text, model_name)
                results[model_name] = result
            except:
                results[model_name] = {"error": "Model not available"}
        
        return {
            "text": text,
            "comparisons": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
