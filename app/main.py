"""
AutoML Healthcare API
A REST API for automated machine learning on healthcare datasets
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

from .ml_pipeline import MLPipeline
from .data_processor import DataProcessor
from .storage import StorageManager

app = FastAPI(
    title="AutoML Healthcare API",
    description="Automated Machine Learning API for Healthcare Datasets",
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

# Initialize components
storage = StorageManager()
data_processor = DataProcessor()
ml_pipeline = MLPipeline()

# In-memory job tracking (use Redis/DB in production)
jobs: Dict[str, Dict[str, Any]] = {}
trained_models: Dict[str, Any] = {}


# ============== Pydantic Models ==============

class TrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    algorithm: Optional[str] = "auto"
    test_size: Optional[float] = 0.2
    
class PredictRequest(BaseModel):
    model_id: str
    data: Dict[str, Any]
    
class BatchPredictRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Any]]

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


# ============== Helper Functions ==============

def create_job(job_type: str) -> str:
    """Create a new job and return job_id"""
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    jobs[job_id] = {
        "job_id": job_id,
        "type": job_type,
        "status": "pending",
        "progress": 0,
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now
    }
    return job_id

def update_job(job_id: str, **kwargs):
    """Update job status"""
    if job_id in jobs:
        jobs[job_id].update(kwargs)
        jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

def get_model_safe(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model from memory or load from storage"""
    if model_id in trained_models:
        return trained_models[model_id]
    
    # Try loading from storage
    model_data = storage.load_model(model_id)
    if model_data:
        trained_models[model_id] = model_data
        return model_data
    
    return None


# ============== Background Tasks ==============

async def train_model_task(job_id: str, dataset_id: str, target_column: str, 
                           algorithm: str, test_size: float):
    """Background task for model training"""
    try:
        update_job(job_id, status="running", progress=10)
        
        # Load dataset
        df = storage.load_dataset(dataset_id)
        if df is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        update_job(job_id, progress=20)
        
        # Preprocess data
        X, y, preprocessor_info = data_processor.preprocess(df, target_column)
        update_job(job_id, progress=40)
        
        # Train model
        model_id = str(uuid.uuid4())
        results = ml_pipeline.train(
            X, y, 
            algorithm=algorithm, 
            test_size=test_size
        )
        update_job(job_id, progress=80)
        
        # Save model
        model_data = {
            "model_id": model_id,
            "model": results["model"],
            "preprocessor_info": preprocessor_info,
            "feature_columns": [col for col in df.columns if col != target_column],
            "target_column": target_column,
            "metrics": results["metrics"],
            "algorithm": results["algorithm_used"],
            "created_at": datetime.utcnow().isoformat()
        }
        trained_models[model_id] = model_data
        storage.save_model(model_id, model_data)
        
        update_job(job_id, status="completed", progress=100, result={
            "model_id": model_id,
            "algorithm": results["algorithm_used"],
            "metrics": results["metrics"],
            "feature_importance": results.get("feature_importance", {})
        })
        
    except Exception as e:
        update_job(job_id, status="failed", error=str(e))


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AutoML Healthcare API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "datasets_count": len(storage.list_datasets()),
        "models_count": len(storage.list_models()),
        "active_jobs": len([j for j in jobs.values() if j["status"] == "running"])
    }


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset for training"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        dataset_id = str(uuid.uuid4())
        
        # Read and validate
        content = await file.read()
        df = data_processor.read_csv(content)
        
        # Basic validation
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")
        
        # Save dataset
        storage.save_dataset(dataset_id, df)
        
        # Generate statistics
        stats = data_processor.get_statistics(df)
        
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "statistics": stats,
            "message": "Dataset uploaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/datasets")
async def list_datasets():
    """List all uploaded datasets"""
    return {
        "datasets": storage.list_datasets()
    }


@app.get("/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get detailed information about a dataset"""
    df = storage.load_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {
        "dataset_id": dataset_id,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "statistics": data_processor.get_statistics(df),
        "sample": df.head(5).to_dict(orient="records")
    }


@app.post("/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start model training job"""
    # Validate dataset exists
    df = storage.load_dataset(request.dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Validate target column
    if request.target_column not in df.columns:
        raise HTTPException(
            status_code=400, 
            detail=f"Target column '{request.target_column}' not found. Available: {list(df.columns)}"
        )
    
    # Create job
    job_id = create_job("training")
    
    # Start background training
    background_tasks.add_task(
        train_model_task,
        job_id,
        request.dataset_id,
        request.target_column,
        request.algorithm,
        request.test_size
    )
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Training job started"
    }


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a training job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


@app.get("/models")
async def list_models():
    """List all trained models"""
    # Use storage metadata to list all models, not just loaded ones
    stored_models = storage.list_models()
    return {
        "models": [
            {
                "model_id": m.get("id"),
                "algorithm": m.get("algorithm"),
                "target_column": m.get("target_column"),
                "metrics": m.get("metrics"),
                "created_at": m.get("created_at")
            }
            for m in stored_models
        ]
    }


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get detailed information about a trained model"""
    model_data = get_model_safe(model_id)
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "model_id": model_data["model_id"],
        "algorithm": model_data["algorithm"],
        "target_column": model_data["target_column"],
        "feature_columns": model_data["feature_columns"],
        "metrics": model_data["metrics"],
        "created_at": model_data["created_at"]
    }


@app.post("/predict")
async def predict(request: PredictRequest):
    """Make a single prediction"""
    model_data = get_model_safe(request.model_id)
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        prediction, probability = ml_pipeline.predict(
            model_data["model"],
            model_data["preprocessor_info"],
            request.data,
            model_data["feature_columns"]
        )
        
        return {
            "prediction": prediction,
            "probability": probability,
            "model_id": request.model_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def batch_predict(request: BatchPredictRequest):
    """Make batch predictions"""
    model_data = get_model_safe(request.model_id)
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        results = []
        
        for record in request.data:
            prediction, probability = ml_pipeline.predict(
                model_data["model"],
                model_data["preprocessor_info"],
                record,
                model_data["feature_columns"]
            )
            results.append({
                "prediction": prediction,
                "probability": probability
            })
        
        return {
            "predictions": results,
            "count": len(results),
            "model_id": request.model_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a trained model"""
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    del trained_models[model_id]
    storage.delete_model(model_id)
    
    return {"message": f"Model {model_id} deleted successfully"}


@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    if not storage.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {"message": f"Dataset {dataset_id} deleted successfully"}