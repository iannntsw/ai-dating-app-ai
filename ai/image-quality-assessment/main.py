from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
import json
from src.evaluater.predict import main as predict_main
from src.handlers.model_builder import Nima
from src.handlers.data_generator import TestDataGenerator
from src.utils.utils import calc_mean_score
import uvicorn

app = FastAPI(title="Image Quality Assessment API", version="1.0.0")

# Global variables for model
nima_model = None
base_model_name = "MobileNet"
weights_file = "models/weights_mobilenet_technical_0.11.hdf5"  # Update this path as needed

@app.on_event("startup")
async def load_model():
    """Load the NIMA model on startup"""
    global nima_model
    try:
        # Initialize the model
        nima = Nima(base_model_name, weights=None)
        nima.build()
        
        # Load weights if they exist
        if os.path.exists(weights_file):
            nima.nima_model.load_weights(weights_file)
            nima_model = nima
            print(f"Model loaded successfully with weights from {weights_file}")
        else:
            print(f"Warning: Weights file {weights_file} not found. Model loaded without pre-trained weights.")
            nima_model = nima
    except Exception as e:
        print(f"Error loading model: {e}")
        nima_model = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Image Quality Assessment API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": nima_model is not None}

@app.post("/predict")
async def predict_image_quality(file: UploadFile = File(...)):
    """Predict image quality score for uploaded image"""
    if nima_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Prepare data for prediction
        img_dir = os.path.dirname(tmp_file_path)
        img_id = os.path.basename(tmp_file_path).split('.')[0]
        samples = [{'image_id': img_id}]
        
        # Initialize data generator
        data_generator = TestDataGenerator(
            samples, 
            img_dir, 
            64, 
            10, 
            nima_model.preprocessing_function(),
            img_format=file.filename.split('.')[-1]
        )
        
        # Get predictions
        predictions = nima_model.nima_model.predict_generator(
            data_generator, 
            workers=1, 
            use_multiprocessing=False, 
            verbose=0
        )
        
        # Calculate mean score
        mean_score = calc_mean_score(predictions[0])
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return {
            "filename": file.filename,
            "mean_score": float(mean_score),
            "prediction": predictions[0].tolist() if len(predictions) > 0 else []
        }
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict-batch")
async def predict_batch_images(files: list[UploadFile] = File(...)):
    """Predict image quality scores for multiple uploaded images"""
    if nima_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    temp_files = []
    
    try:
        for file in files:
            # Validate file type
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "error": "File must be an image"
                })
                continue
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
                temp_files.append(tmp_file_path)
            
            # Prepare data for prediction
            img_dir = os.path.dirname(tmp_file_path)
            img_id = os.path.basename(tmp_file_path).split('.')[0]
            samples = [{'image_id': img_id}]
            
            # Initialize data generator
            data_generator = TestDataGenerator(
                samples, 
                img_dir, 
                64, 
                10, 
                nima_model.preprocessing_function(),
                img_format=file.filename.split('.')[-1]
            )
            
            # Get predictions
            predictions = nima_model.nima_model.predict_generator(
                data_generator, 
                workers=1, 
                use_multiprocessing=False, 
                verbose=0
            )
            
            # Calculate mean score
            mean_score = calc_mean_score(predictions[0])
            
            results.append({
                "filename": file.filename,
                "mean_score": float(mean_score),
                "prediction": predictions[0].tolist() if len(predictions) > 0 else []
            })
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")
    
    finally:
        # Clean up temporary files
        for tmp_file in temp_files:
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)
