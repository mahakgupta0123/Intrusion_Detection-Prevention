"""
Prevention API - FastAPI Server

Serve prevention models via REST API with web dashboard.
"""

from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import os
import io
import json
import numpy as np

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from pathlib import Path


# Paths
MODELS_DIR = 'trained_models'
TEMPLATES_DIR = 'templates'
DATA_DIR = 'data/processed'

# Attack type explanations
ATTACK_EXPLANATIONS = {
    'Normal': 'No signs of attack detected. Network traffic is safe.',
    'DoS': 'Denial of Service attack caused by overwhelming traffic or protocol abuse.',
    'Exploit': 'Leverages software vulnerabilities to gain unauthorized access.',
    'Worm': 'Self-replicating attack spreading through the network.',
    'Fuzzer': 'Random/malformed data injection to crash or exploit systems.',
    'Generic': 'Pattern-matching attack commonly flagged by IDS tools.',
    'Reconnaissance': 'Probing or scanning activities to discover system weaknesses.'
}


def create_app(model_path=None, scaler_path=None):
    """
    Create FastAPI application for prevention.
    
    Args:
        model_path: Path to trained model
        scaler_path: Path to scaler
        
    Returns:
        FastAPI application
    """
    app = FastAPI(title="IDS/IPS Prevention API", version="1.0.0")
    
    # Load model
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, 'ids_model_rf_augmented.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    
    # Load scaler if available
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    # Load templates if available
    if os.path.exists(TEMPLATES_DIR):
        templates = Jinja2Templates(directory=TEMPLATES_DIR)
    else:
        templates = None
    
    @app.get("/", response_class=HTMLResponse)
    async def read_form(request: Request):
        """Serve dashboard."""
        if templates:
            return templates.TemplateResponse("dashboard.html", {"request": request})
        else:
            return """
            <html>
                <body>
                    <h1>IDS/IPS Prevention API</h1>
                    <p>Use POST /analyze endpoint to analyze traffic</p>
                    <p>Use GET /docs for API documentation</p>
                </body>
            </html>
            """
    
    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...)):
        """
        Analyze uploaded CSV file.
        
        Args:
            file: CSV file with network traffic data
            
        Returns:
            JSON with predictions and analysis
        """
        try:
            contents = await file.read()
            try:
                df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            except UnicodeDecodeError:
                df = pd.read_csv(io.StringIO(contents.decode('ISO-8859-1')))
            
            # Ensure model features are present
            if model_features is not None:
                if not all(f in df.columns for f in model_features):
                    missing = [f for f in model_features if f not in df.columns]
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing features: {missing}"
                    )
                X = df[model_features]
            else:
                X = df
            
            # Scale if scaler available
            if scaler:
                X = scaler.transform(X)
            
            # Make predictions
            predictions = model.predict(X)
            probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            
            # Prepare response
            results = {
                'total_samples': len(df),
                'predictions': predictions.tolist(),
                'attack_count': int((predictions == 1).sum()),
                'normal_count': int((predictions == 0).sum())
            }
            
            if probabilities is not None:
                results['probabilities'] = probabilities.tolist()
            
            return results
        
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/predict")
    async def predict(data: dict):
        """
        Make prediction on input features.
        
        Args:
            data: Dictionary with features
            
        Returns:
            Prediction result
        """
        try:
            df = pd.DataFrame([data])
            
            # Ensure all required features are present
            if model_features is not None:
                if not all(f in df.columns for f in model_features):
                    missing = [f for f in model_features if f not in df.columns]
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing features: {missing}"
                    )
                X = df[model_features]
            else:
                X = df
            
            # Scale if available
            if scaler:
                X = scaler.transform(X)
            
            # Predict
            prediction = model.predict(X)[0]
            probability = None
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                probability = float(proba[1])  # Attack probability
            
            return {
                'prediction': int(prediction),
                'prediction_label': 'Attack' if prediction == 1 else 'Normal',
                'probability': probability,
                'confidence': max(probability, 1 - probability) if probability else None
            }
        
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            'status': 'healthy',
            'model_type': type(model).__name__,
            'features': len(model_features) if model_features else 'unknown'
        }
    
    @app.get("/stats")
    async def stats():
        """Get model statistics."""
        return {
            'model_type': type(model).__name__,
            'num_features': len(model_features) if model_features else 'unknown',
            'has_scaler': scaler is not None,
            'has_shap': HAS_SHAP
        }
    
    return app


def run_server(model_path=None, host='0.0.0.0', port=8000):
    """
    Run FastAPI prevention server.
    
    Args:
        model_path: Path to model
        host: Host to bind to
        port: Port to bind to
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required to run server. Install with: pip install uvicorn")
    
    app = create_app(model_path)
    uvicorn.run(app, host=host, port=port)
