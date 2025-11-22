"""
Day 26: Assignment - Deploy ML Model as REST API using FastAPI
Complete production-ready deployment example
"""

import os
import joblib
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import json

# FastAPI
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn")

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# ==================== PYDANTIC MODELS ====================

if FASTAPI_AVAILABLE:
    class PredictionInput(BaseModel):
        """Input schema for predictions"""
        features: List[float] = Field(..., description="List of feature values")

        class Config:
            json_schema_extra = {
                "example": {
                    "features": [0.5, -0.3, 1.2, 0.8, -0.5, 0.1, 0.9, -0.2,
                                0.4, 0.6, -0.1, 0.3, 0.7, -0.4, 0.2, 0.5,
                                -0.3, 0.8, 0.1, -0.6]
                }
            }

    class BatchPredictionInput(BaseModel):
        """Input schema for batch predictions"""
        instances: List[List[float]] = Field(..., description="List of feature vectors")

    class PredictionOutput(BaseModel):
        """Output schema for predictions"""
        prediction: int
        probability: List[float]
        model_version: str
        timestamp: str

    class BatchPredictionOutput(BaseModel):
        """Output schema for batch predictions"""
        predictions: List[int]
        probabilities: List[List[float]]
        model_version: str
        timestamp: str
        count: int

    class ModelInfo(BaseModel):
        """Model information schema"""
        name: str
        version: str
        description: str
        features_expected: int
        classes: List[int]
        metrics: Dict[str, float]

    class HealthResponse(BaseModel):
        """Health check response"""
        status: str
        model_loaded: bool
        timestamp: str


# ==================== MODEL MANAGER ====================

class ModelManager:
    """
    Manages ML model lifecycle: loading, versioning, predictions
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.model_info = {}
        self.prediction_count = 0

        os.makedirs(model_dir, exist_ok=True)

    def train_demo_model(self):
        """Train a demo model for testing"""
        print("Training demo model...")

        # Generate synthetic data
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_classes=2,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create and train pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        pipeline.fit(X_train, y_train)
        accuracy = pipeline.score(X_test, y_test)

        # Save model
        model_path = os.path.join(self.model_dir, "model.joblib")
        joblib.dump(pipeline, model_path)

        # Save model info
        self.model_info = {
            'name': 'RandomForestClassifier',
            'version': '1.0.0',
            'description': 'Binary classification model',
            'features_expected': 20,
            'classes': [0, 1],
            'metrics': {'accuracy': float(accuracy)},
            'trained_at': datetime.now().isoformat()
        }

        info_path = os.path.join(self.model_dir, "model_info.json")
        with open(info_path, 'w') as f:
            json.dump(self.model_info, f, indent=2)

        print(f"Model trained with accuracy: {accuracy:.4f}")
        print(f"Model saved to: {model_path}")

        return pipeline

    def load_model(self):
        """Load model from disk"""
        model_path = os.path.join(self.model_dir, "model.joblib")
        info_path = os.path.join(self.model_dir, "model_info.json")

        if not os.path.exists(model_path):
            # Train demo model if not exists
            self.model = self.train_demo_model()
        else:
            self.model = joblib.load(model_path)
            print(f"Model loaded from: {model_path}")

        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.model_info = json.load(f)

    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Make single prediction"""
        if self.model is None:
            raise ValueError("Model not loaded")

        X = np.array(features).reshape(1, -1)

        # Get prediction and probabilities
        prediction = int(self.model.predict(X)[0])
        probability = self.model.predict_proba(X)[0].tolist()

        self.prediction_count += 1

        return {
            'prediction': prediction,
            'probability': probability,
            'model_version': self.model_info.get('version', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }

    def predict_batch(self, instances: List[List[float]]) -> Dict[str, Any]:
        """Make batch predictions"""
        if self.model is None:
            raise ValueError("Model not loaded")

        X = np.array(instances)

        predictions = self.model.predict(X).tolist()
        probabilities = self.model.predict_proba(X).tolist()

        self.prediction_count += len(instances)

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'model_version': self.model_info.get('version', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'count': len(instances)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_info


# ==================== FASTAPI APPLICATION ====================

# Initialize
model_manager = ModelManager()

if FASTAPI_AVAILABLE:
    # Create FastAPI app
    app = FastAPI(
        title="ML Model API",
        description="REST API for ML model predictions",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    # ==================== STARTUP/SHUTDOWN ====================

    @app.on_event("startup")
    async def startup_event():
        """Load model on startup"""
        logger.info("Starting up API...")
        model_manager.load_model()
        logger.info("Model loaded successfully")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        logger.info("Shutting down API...")
        logger.info(f"Total predictions made: {model_manager.prediction_count}")


    # ==================== ENDPOINTS ====================

    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "ML Model API",
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            model_loaded=model_manager.model is not None,
            timestamp=datetime.now().isoformat()
        )

    @app.get("/model/info", response_model=ModelInfo, tags=["Model"])
    async def get_model_info():
        """Get model information"""
        info = model_manager.get_model_info()
        if not info:
            raise HTTPException(status_code=404, detail="Model info not found")
        return ModelInfo(**info)

    @app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
    async def predict(input_data: PredictionInput):
        """
        Make a single prediction

        - **features**: List of 20 numerical features
        """
        try:
            # Validate input
            if len(input_data.features) != model_manager.model_info.get('features_expected', 20):
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected {model_manager.model_info.get('features_expected', 20)} features, "
                           f"got {len(input_data.features)}"
                )

            result = model_manager.predict(input_data.features)
            return PredictionOutput(**result)

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Predictions"])
    async def predict_batch(input_data: BatchPredictionInput):
        """
        Make batch predictions

        - **instances**: List of feature vectors
        """
        try:
            if not input_data.instances:
                raise HTTPException(status_code=400, detail="No instances provided")

            # Validate all instances
            expected_features = model_manager.model_info.get('features_expected', 20)
            for i, instance in enumerate(input_data.instances):
                if len(instance) != expected_features:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Instance {i}: Expected {expected_features} features, got {len(instance)}"
                    )

            result = model_manager.predict_batch(input_data.instances)
            return BatchPredictionOutput(**result)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/stats", tags=["Monitoring"])
    async def get_stats():
        """Get API statistics"""
        return {
            "total_predictions": model_manager.prediction_count,
            "model_version": model_manager.model_info.get('version', 'unknown'),
            "timestamp": datetime.now().isoformat()
        }


# ==================== FLASK ALTERNATIVE ====================

def create_flask_app():
    """
    Alternative Flask implementation
    """
    try:
        from flask import Flask, request, jsonify
        FLASK_AVAILABLE = True
    except ImportError:
        print("Flask not available. Install with: pip install flask")
        return None

    flask_app = Flask(__name__)

    @flask_app.route('/', methods=['GET'])
    def flask_root():
        return jsonify({"message": "ML Model API (Flask)"})

    @flask_app.route('/health', methods=['GET'])
    def flask_health():
        return jsonify({
            "status": "healthy",
            "model_loaded": model_manager.model is not None,
            "timestamp": datetime.now().isoformat()
        })

    @flask_app.route('/predict', methods=['POST'])
    def flask_predict():
        try:
            data = request.get_json()
            features = data.get('features', [])

            if not features:
                return jsonify({"error": "No features provided"}), 400

            result = model_manager.predict(features)
            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return flask_app


# ==================== MAIN ====================

def demonstrate_api():
    """Demonstrate the API without running server"""
    print("=" * 60)
    print("ML MODEL REST API DEMONSTRATION")
    print("=" * 60)

    # Initialize model
    model_manager.load_model()

    # Test prediction
    print("\n1. Testing single prediction:")
    test_features = np.random.randn(20).tolist()
    result = model_manager.predict(test_features)
    print(f"   Input: {test_features[:5]}... (20 features)")
    print(f"   Prediction: {result['prediction']}")
    print(f"   Probability: {result['probability']}")

    # Test batch prediction
    print("\n2. Testing batch prediction:")
    test_batch = np.random.randn(5, 20).tolist()
    result = model_manager.predict_batch(test_batch)
    print(f"   Batch size: {result['count']}")
    print(f"   Predictions: {result['predictions']}")

    # Model info
    print("\n3. Model information:")
    info = model_manager.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("TO RUN THE API:")
    print("=" * 60)
    print("""
    # FastAPI (recommended):
    uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000

    # Then access:
    # - API docs: http://localhost:8000/docs
    # - Health:   http://localhost:8000/health
    # - Predict:  POST http://localhost:8000/predict

    # Example curl command:
    curl -X POST "http://localhost:8000/predict" \\
         -H "Content-Type: application/json" \\
         -d '{"features": [0.5, -0.3, 1.2, 0.8, -0.5, 0.1, 0.9, -0.2, 0.4, 0.6, -0.1, 0.3, 0.7, -0.4, 0.2, 0.5, -0.3, 0.8, 0.1, -0.6]}'
    """)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # Run the server
        if FASTAPI_AVAILABLE:
            uvicorn.run(app, host="0.0.0.0", port=8000)
        else:
            print("FastAPI not available")
    else:
        # Demo mode
        demonstrate_api()
