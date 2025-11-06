from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai.profile_management.ai_profile_management import init_ai
from ai.ai_lovabot.ai_lovabot import chat as lovabot_chat
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
import os
import subprocess
import json
import tempfile
import requests
from urllib.parse import urlparse
import base64
import numpy as np
import schedule
import time
import threading
import uvicorn

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.ai_date_planner.ai_date_planner import AIDatePlanner
from ai.ai_date_planner.rule_engine import UserPreferences
from ai.ai_date_planner.vendor_embedding_service import VendorEmbeddingService
from ai.conversation_starters.ai_conversation_starters import generate_conversation_starters
# Use InsightFace-based verification AI for advanced face analysis
from ai.verification.insightface_verification_ai import insightface_verification_ai as verification_ai
from ai.intro_ai.intro_ai import generate_introduction_from_request

# --- AI Re-ranker imports ---
from ai.discover_profiles.models import Payload
from ai.discover_profiles.ranking import rank_recommendations
from ai.discover_profiles.packs import (
    PackRankingRequest,
    rank_packs as rank_packs_fn,
    generate_fun_pack_name,
)

app = FastAPI()

# Initialize the AI Date Planner
planner = None

@app.on_event("startup")
async def startup_event():
    """Initialize the AI Date Planner on startup"""
    global planner
    try:
        planner = AIDatePlanner(data_dir="ai/ai_date_planner/data")
        print("AI Date Planner initialized successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Failed to initialize AI Date Planner: {e}")
        print("⚠️  Date planning features will be unavailable, but the server will continue to run")
        planner = None
        # Don't raise - allow server to start even if Date Planner fails
    
    # Start cron scheduler when the app starts
    start_cron_scheduler()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def get_hello():
    """Health check endpoint - Render uses this to detect the service"""
    return {"status": "ok", "message": "AI Service is running"}

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "planner_ready": planner is not None}
    

@app.post("/ai/bio")
def create_bio(bio_interests: list[str]) -> list[str]:
    model = init_ai()
    prompt =""
    with open("ai/profile_management/ai_bio_generator.md", "r") as f:
        prompt = f.read()
    system_prompt = SystemMessage(content=prompt)
    interest =""
    interest = ", ".join(bio_interests)
    human_prompt = HumanMessage(content=interest)
    response = model.invoke([system_prompt, human_prompt])
    
    # Split the response into individual bios and return as list
    bios = response.content.split('\n\n')
    cleaned_bios = []
    for bio in bios:
        if bio.strip():
            if ':' in bio:
                bio = bio.split(':', 1)[1].strip()
            cleaned_bios.append(bio.strip())
    
    return cleaned_bios

@app.post("/ai/prompts")
def generate_prompt_response(request: dict) -> str:
    model = init_ai()
    
    # Read the prompt generator instructions
    with open("ai/profile_management/ai_prompt_generator.md", "r") as f:
        system_instructions = f.read()
    
    # Extract question and user's answer from request
    question = request.get("question", "")
    user_answer = request.get("answer", "")
    
    # Create the system prompt
    system_prompt = SystemMessage(content=system_instructions)
    
    # Create the human prompt with context
    human_prompt_content = f"""
Question: {question}

User's current answer: {user_answer}

Please generate an enhanced version of the user's answer that is more engaging and authentic while maintaining their original intent and personality.
"""
    
    human_prompt = HumanMessage(content=human_prompt_content)
    
    # Get AI response
    response = model.invoke([system_prompt, human_prompt])
    
    return response.content.strip()

@app.post("/ai/introduction")
def generate_introduction_message(request: dict):
    """
    Generate AI-powered introduction message based on context (bio, interests, or prompt).
    
    Request format:
    {
        "type": "bio" | "interests" | "prompt" | "general",
        "name": "John",
        "bio": "...",  // for type="bio"
        "interests": ["hiking", "photography"],  // for type="interests"
        "question": "What's your ideal weekend?",  // for type="prompt"
        "answer": "Exploring new hiking trails"  // for type="prompt"
    }
    
    Response format:
    {
        "message": "Generated introduction message..."
    }
    """
    model = init_ai()
    return generate_introduction_from_request(model, request)

@app.post("/ai/lovabot")
def generate_lovabot_response(request: dict):
    # Extract messages from request
    messages = request.get("messages", [])
    
    if not messages:
        return {"answer": "No messages provided."}
    
    # Use lovabot chat function with RAG
    response = lovabot_chat(messages)
    return response

@app.post("/ai/generate-conversation-starters")
def generate_conversation_starters_endpoint(request: dict):
    """
    Generate personalized conversation starters for two users.
    
    Request format:
    {
        "user1": {
            "name": "John",
            "age": 25,
            "gender": "male",
            "interests": ["photography", "hiking"],
            "bio": "Love outdoor adventures",
            "job": "Photographer",
            "education": "College",
            "location": "New York"
        },
        "user2": {
            "name": "Sarah", 
            "age": 23,
            "gender": "female",
            "interests": ["photography", "coffee"],
            "bio": "Coffee enthusiast and photographer",
            "job": "Designer",
            "education": "University",
            "location": "New York"
        },
        "sharedInterests": ["photography"]
    }
    
    Response format:
    {
        "success": true,
        "starters": [
            {
                "text": "I see we both love photography! What got you into it?",
                "type": "interest",
                "category": "photography", 
                "confidence": 0.9
            }
        ]
    }
    """
    try:
        # Extract data from request
        user1 = request.get("user1", {})
        user2 = request.get("user2", {})
        shared_interests = request.get("sharedInterests", [])
        
        # Generate conversation starters (support refresh flag to bypass cache)
        refresh = bool(request.get("refresh", False))
        starters = generate_conversation_starters(user1, user2, shared_interests, refresh)
        
        return {
            "success": True,
            "starters": starters
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error generating conversation starters: {str(e)}",
            "starters": []
        }

# Date Planner Models
class DatePlanRequest(BaseModel):
    start_time: str = "10:00"
    end_time: Optional[str] = None
    date: Optional[str] = None  # Date in YYYY-MM-DD format (defaults to today if not provided)
    start_latitude: Optional[float] = None
    start_longitude: Optional[float] = None
    interests: List[str] = ["food", "culture", "nature"]
    budget_tier: str = "$$"
    date_type: str = "casual"
    exclusions: Optional[List[str]] = None  # Backend-only: What user does NOT want

# Image Quality Assessment Models
class ImageQualityRequest(BaseModel):
    image_urls: Optional[List[str]] = []
    image_data: Optional[List[str]] = []  # Base64 encoded images

class ImageQualityResult(BaseModel):
    image_url: str
    technical_score: float
    aesthetic_score: float
    aggregate_score: float

class ImageQualityResponse(BaseModel):
    results: List[ImageQualityResult]
    top_recommendations: List[str]

@app.post("/ai/plan-date")
def plan_date(request: DatePlanRequest):
    """Plan a date based on user preferences"""
    if planner is None:
        return {"error": "AI Date Planner not initialized"}
    
    try:
        # Convert request to UserPreferences
        preferences = UserPreferences(
            start_time=request.start_time,
            end_time=request.end_time,
            date=request.date,
            start_latitude=request.start_latitude,
            start_longitude=request.start_longitude,
            interests=request.interests,
            budget_tier=request.budget_tier,
            date_type=request.date_type
        )
        
        # Plan the date with exclusions
        result = planner.plan_date(preferences, request.exclusions)
        
        return {
            "success": True,
            "itinerary": result.date_plan.itinerary,
            "total_duration": result.date_plan.total_duration,
            "estimated_cost": result.date_plan.estimated_cost,
            "summary": result.date_plan.summary,
            "alternative_suggestions": result.date_plan.alternative_suggestions,
            "processing_stats": result.processing_stats
        }
        
    except Exception as e:
        return {"error": f"Error planning date: {str(e)}"}

    
# ==============================
# AI Re-ranker
# ==============================

@app.post("/rank/recommendations")
def rank_recommendations_endpoint(payload: Payload):
    """Rank candidate profiles based on compatibility with user."""
    return rank_recommendations(payload)


class PackNameRequest(BaseModel):
    interest: str
    category: Optional[str] = None
    use_llm: bool = True


@app.post("/rank/packs")
def rank_packs_endpoint(req: PackRankingRequest):
    """Rank interest-based packs by relevance to the user's interests."""
    return [r.dict() for r in rank_packs_fn(req)]


@app.post("/generate/pack-name")
def generate_pack_name_endpoint(req: PackNameRequest):
    """Generate a fun 2–3 word pack name for an interest."""
    model = init_ai() if req.use_llm else None
    name = generate_fun_pack_name(req.interest, req.category, model=model)
    return {"name": name}
# ==============================
# Image Quality Assessment
# ==============================

def download_image_to_temp(image_url: str) -> str:
    """Download image from URL to temporary file and return the path."""
    try:
        print("FastAPI: Downloading image from:", image_url)
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        print("FastAPI: Image downloaded successfully, size:", len(response.content), "bytes")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(response.content)
            print("FastAPI: Created temporary file:", temp_file.name)
            return temp_file.name
    except Exception as e:
        print("FastAPI: Error downloading image", image_url, ":", e)
        raise

def save_base64_image_to_temp(base64_data: str) -> str:
    """Save base64 encoded image to temporary file and return the path."""
    try:
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        # Decode base64 data
        image_data = base64.b64decode(base64_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(image_data)
            print("FastAPI: Created temporary file from base64:", temp_file.name)
            return temp_file.name
    except Exception as e:
        print("FastAPI: Error saving base64 image:", e)
        raise

def assess_image_quality(image_path: str) -> dict:
    """Assess image quality using NIMA Docker container."""
    try:
        # Get absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        technical_weights = os.path.join(current_dir, "ai/image-quality-assessment/models/MobileNet/weights_mobilenet_technical_0.11.hdf5")
        aesthetic_weights = os.path.join(current_dir, "ai/image-quality-assessment/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5")
        
        # Check if files exist
        if not os.path.exists(technical_weights):
            raise Exception(f"Technical weights file not found: {technical_weights}")
        if not os.path.exists(aesthetic_weights):
            raise Exception(f"Aesthetic weights file not found: {aesthetic_weights}")
        if not os.path.exists(image_path):
            raise Exception(f"Image file not found: {image_path}")
        
        # Convert Windows paths to Docker-compatible format
        def to_docker_path(path):
            # Convert Windows path to Docker format
            path = path.replace('\\', '/')
            if path.startswith('C:'):
                path = path.replace('C:', '/c', 1)
            return path
        
        docker_image_path = to_docker_path(image_path)
        docker_technical_weights = to_docker_path(technical_weights)
        docker_aesthetic_weights = to_docker_path(aesthetic_weights)
        
        # Run technical assessment
        technical_cmd = [
            "docker", "run", "--rm", "--entrypoint", "",
            "-v", f"{docker_image_path}:/src/image.jpg",
            "-v", f"{docker_technical_weights}:/src/weights.hdf5",
            "nima-cpu", "python", "-m", "evaluater.predict",
            "--base-model-name", "MobileNet",
            "--weights-file", "/src/weights.hdf5",
            "--image-source", "/src/image.jpg"
        ]
        
        technical_result = subprocess.run(technical_cmd, capture_output=True, text=True, timeout=60)
        
        if technical_result.returncode != 0:
            raise Exception(f"Technical assessment failed: {technical_result.stderr}")
        
        # Clean the stdout to extract only the JSON part
        stdout_lines = technical_result.stdout.strip().split('\n')
        json_lines = []
        in_json = False
        
        for line in stdout_lines:
            if line.strip().startswith('['):
                in_json = True
            if in_json:
                json_lines.append(line)
            if in_json and line.strip().endswith(']'):
                break
        
        if not json_lines:
            raise Exception("No JSON output found in technical assessment result")
        
        json_str = ''.join(json_lines)
        technical_score = json.loads(json_str)[0]["mean_score_prediction"]
        
        # Run aesthetic assessment
        aesthetic_cmd = [
            "docker", "run", "--rm", "--entrypoint", "",
            "-v", f"{docker_image_path}:/src/image.jpg",
            "-v", f"{docker_aesthetic_weights}:/src/weights.hdf5",
            "nima-cpu", "python", "-m", "evaluater.predict",
            "--base-model-name", "MobileNet",
            "--weights-file", "/src/weights.hdf5",
            "--image-source", "/src/image.jpg"
        ]
        
        aesthetic_result = subprocess.run(aesthetic_cmd, capture_output=True, text=True, timeout=60)
        
        if aesthetic_result.returncode != 0:
            raise Exception(f"Aesthetic assessment failed: {aesthetic_result.stderr}")
        
        # Clean the stdout to extract only the JSON part
        stdout_lines = aesthetic_result.stdout.strip().split('\n')
        json_lines = []
        in_json = False
        
        for line in stdout_lines:
            if line.strip().startswith('['):
                in_json = True
            if in_json:
                json_lines.append(line)
            if in_json and line.strip().endswith(']'):
                break
        
        if not json_lines:
            raise Exception("No JSON output found in aesthetic assessment result")
        
        json_str = ''.join(json_lines)
        aesthetic_score = json.loads(json_str)[0]["mean_score_prediction"]
        
        # Calculate aggregate score (average of technical and aesthetic)
        aggregate_score = (technical_score + aesthetic_score) / 2
        
        result = {
            "technical_score": technical_score,
            "aesthetic_score": aesthetic_score,
            "aggregate_score": aggregate_score
        }
        
        return result
        
    except Exception as e:
        # Return default scores if assessment fails
        return {
            "technical_score": 5.0,
            "aesthetic_score": 5.0,
            "aggregate_score": 5.0
        }

import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

def assess_single_image_quality(image_data: tuple) -> ImageQualityResult:
    """Assess quality of a single image - designed for parallel execution."""
    try:
        if isinstance(image_data, tuple) and len(image_data) == 2:
            # Base64 image
            i, base64_data = image_data
            temp_path = save_base64_image_to_temp(base64_data)
            image_url = f"base64_image_{i}"
        else:
            # URL image
            image_url = image_data
            temp_path = download_image_to_temp(image_url)
        
        # Assess image quality
        quality_scores = assess_image_quality(temp_path)
        
        # Clean up temp file immediately
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return ImageQualityResult(
            image_url=image_url,
            technical_score=quality_scores["technical_score"],
            aesthetic_score=quality_scores["aesthetic_score"],
            aggregate_score=quality_scores["aggregate_score"]
        )
    except Exception as e:
        print(f"Error assessing image {image_data}: {e}")
        # Return fallback result
        image_url = f"base64_image_{image_data[0]}" if isinstance(image_data, tuple) else image_data
        return ImageQualityResult(
            image_url=image_url,
            technical_score=5.0,
            aesthetic_score=5.0,
            aggregate_score=5.0
        )

@app.post("/ai/assess-image-quality", response_model=ImageQualityResponse)
def assess_image_quality_endpoint(request: ImageQualityRequest):
    """Assess the quality of multiple images and return top recommendations."""
    try:
        # Prepare all images for parallel processing
        images_to_process = []
        
        # Add base64 images
        if request.image_data:
            for i, base64_data in enumerate(request.image_data):
                images_to_process.append((i, base64_data))
        
        # Add URL images
        if request.image_urls:
            for image_url in request.image_urls:
                images_to_process.append(image_url)
        
        if not images_to_process:
            return ImageQualityResponse(results=[], top_recommendations=[])
        
        # Process all images in parallel using ThreadPoolExecutor
        print(f"Processing {len(images_to_process)} images in parallel...")
        with ThreadPoolExecutor(max_workers=len(images_to_process)) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(assess_single_image_quality, image_data): image_data 
                for image_data in images_to_process
            }
            
            # Collect results as they complete
            results = []
            for future in concurrent.futures.as_completed(future_to_image):
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Completed assessment for {result.image_url}: {result.aggregate_score}")
                except Exception as e:
                    print(f"Error processing image: {e}")
                    # Add fallback result
                    image_data = future_to_image[future]
                    image_url = f"base64_image_{image_data[0]}" if isinstance(image_data, tuple) else image_data
                    results.append(ImageQualityResult(
                        image_url=image_url,
                        technical_score=5.0,
                        aesthetic_score=5.0,
                        aggregate_score=5.0
                    ))
        
        # Sort by aggregate score (highest first) and get top 2
        sorted_results = sorted(results, key=lambda x: x.aggregate_score, reverse=True)
        top_recommendations = [result.image_url for result in sorted_results[:2]]
        
        print(f"Assessment complete. Top recommendations: {top_recommendations}")
        
        response = ImageQualityResponse(
            results=results,
            top_recommendations=top_recommendations
        )
        
        return response
        
    except Exception as e:
        print(f"Error in assess_image_quality_endpoint: {e}")
        # Return fallback response
        all_images = (request.image_urls or []) + [f"base64_image_{i}" for i in range(len(request.image_data or []))]
        fallback_results = [
            ImageQualityResult(
                image_url=url,
                technical_score=5.0,
                aesthetic_score=5.0,
                aggregate_score=5.0
            ) for url in all_images
        ]
        
        fallback_response = ImageQualityResponse(
            results=fallback_results,
            top_recommendations=all_images[:2]
        )
        
        return fallback_response

# ==============================
# AI Verification System
# ==============================

class VerificationRequest(BaseModel):
    photo_urls: List[str]
    user_id: str
    verification_level: str = "basic"

class VerificationResponse(BaseModel):
    success: bool
    verification_id: str
    overall_assessment: Dict[str, Any]
    processing_time: float

@app.post("/ai/verify-user", response_model=VerificationResponse)
def verify_user_endpoint(request: VerificationRequest):
    """
    Comprehensive user verification using AI analysis
    """
    import time
    start_time = time.time()
    
    try:
        print(f"Starting verification for user {request.user_id} with {len(request.photo_urls)} photos")
        
        # Perform AI analysis
        analysis_results = verification_ai.analyze_verification_photos(
            request.photo_urls, 
            request.user_id
        )
        
        # Extract overall assessment
        overall_assessment = analysis_results['overall_assessment']
        
        # Generate verification ID
        verification_id = f"verification_{request.user_id}_{int(time.time())}"
        
        processing_time = time.time() - start_time
        
        return VerificationResponse(
            success=True,
            verification_id=verification_id,
            overall_assessment=overall_assessment,
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        print(f"Error in verification: {e}")
        processing_time = time.time() - start_time
        
        return VerificationResponse(
            success=False,
            verification_id=f"error_{int(time.time())}",
            overall_assessment={
                'overall_score': 0.0,
                'verification_status': 'error',
                'requires_manual_review': True,
                'recommendations': [f"Verification failed: {str(e)}"],
                'confidence_level': 'low',
                'face_consistency_score': 0.0,
                'photo_quality_score': 0.0,
                'deepfake_risk_score': 100.0
            },
            processing_time=round(processing_time, 2)
        )

class FaceAnalysisRequest(BaseModel):
    photo_urls: List[str]

class FaceAnalysisResponse(BaseModel):
    success: bool
    face_analyses: List[Dict[str, Any]]
    consistency_score: float
    face_count_per_photo: List[int]

@app.post("/ai/analyze-faces", response_model=FaceAnalysisResponse)
def analyze_faces_endpoint(request: FaceAnalysisRequest):
    """
    Analyze faces in verification photos
    """
    try:
        print(f"Analyzing faces in {len(request.photo_urls)} photos")
        
        face_analyses = []
        face_counts = []
        
        for photo_url in request.photo_urls:
            # Download image
            image_data = verification_ai._download_image(photo_url)
            if image_data is not None:
                # Analyze faces
                face_analysis = verification_ai._analyze_faces_insightface(image_data)
                face_analyses.append(face_analysis)
                face_counts.append(face_analysis['face_count'])
            else:
                face_analyses.append({
                    'face_count': 0,
                    'face_detected': False,
                    'error': 'Failed to download image'
                })
                face_counts.append(0)
        
        # Calculate consistency if multiple photos
        consistency_score = 0
        if len(face_analyses) > 1:
            consistency_result = verification_ai._calculate_face_consistency(face_analyses)
            consistency_score = consistency_result['consistency_score']
        
        return FaceAnalysisResponse(
            success=True,
            face_analyses=face_analyses,
            consistency_score=consistency_score,
            face_count_per_photo=face_counts
        )
        
    except Exception as e:
        print(f"Error in face analysis: {e}")
        return FaceAnalysisResponse(
            success=False,
            face_analyses=[],
            consistency_score=0.0,
            face_count_per_photo=[]
        )

class DeepfakeDetectionRequest(BaseModel):
    photo_urls: List[str]

class DeepfakeDetectionResponse(BaseModel):
    success: bool
    deepfake_analyses: List[Dict[str, Any]]
    overall_risk_score: float
    high_risk_photos: List[int]

@app.post("/ai/detect-deepfake", response_model=DeepfakeDetectionResponse)
def detect_deepfake_endpoint(request: DeepfakeDetectionRequest):
    """
    Detect potential deepfake or manipulated images
    

    """
    try:
        print(f"Detecting deepfakes in {len(request.photo_urls)} photos")
        
        deepfake_analyses = []
        high_risk_photos = []
        
        for i, photo_url in enumerate(request.photo_urls):
            # Download image
            image_data = verification_ai._download_image(photo_url)
            if image_data is not None:
                # Detect deepfake
                deepfake_analysis = verification_ai._detect_deepfake(image_data)
                deepfake_analyses.append(deepfake_analysis)
                
                if deepfake_analysis['manipulation_detected']:
                    high_risk_photos.append(i)
            else:
                deepfake_analyses.append({
                    'deepfake_probability': 0.0,
                    'manipulation_detected': False,
                    'error': 'Failed to download image'
                })
        
        # Calculate overall risk score
        risk_scores = [analysis.get('deepfake_probability', 0) for analysis in deepfake_analyses]
        overall_risk_score = np.mean(risk_scores) * 100 if risk_scores else 0
        
        return DeepfakeDetectionResponse(
            success=True,
            deepfake_analyses=deepfake_analyses,
            overall_risk_score=round(overall_risk_score, 2),
            high_risk_photos=high_risk_photos
        )
        
    except Exception as e:
        print(f"Error in deepfake detection: {e}")
        return DeepfakeDetectionResponse(
            success=False,
            deepfake_analyses=[],
            overall_risk_score=100.0,
            high_risk_photos=[]
        )

# ============================================================================
# VENDOR EMBEDDING ENDPOINTS
# ============================================================================

class VendorEmbeddingResponse(BaseModel):
    success: bool
    message: str
    vendor_count: int = 0
    embeddings_generated: bool = False
    faiss_index_built: bool = False

@app.post("/api/vendor/embeddings/cron", response_model=VendorEmbeddingResponse)
async def regenerate_vendor_embeddings_cron():
    """Cron job endpoint to regenerate vendor embeddings daily at 12 AM"""
    try:
        print("🔄 CRON: Starting vendor embedding regeneration...")
        
        # Initialize vendor embedding service
        vendor_service = VendorEmbeddingService()
        
        # Generate vendor embeddings
        embeddings = vendor_service.generate_vendor_embeddings(force_regenerate=True)
        
        if embeddings.size == 0:
            return VendorEmbeddingResponse(
                success=True,
                message="No vendor activities found to process",
                vendor_count=0,
                embeddings_generated=False,
                faiss_index_built=False
            )
        
        # Build FAISS index
        faiss_index = vendor_service.build_vendor_faiss_index()
        
        # Disconnect from MongoDB
        vendor_service.disconnect_from_mongodb()
        
        print("✅ CRON: Vendor embedding regeneration completed successfully")
        
        return VendorEmbeddingResponse(
            success=True,
            message="Vendor embeddings regenerated successfully via cron",
            vendor_count=len(vendor_service.vendor_locations),
            embeddings_generated=True,
            faiss_index_built=True
        )
        
    except Exception as e:
        print(f"❌ CRON: Error regenerating vendor embeddings: {e}")
        return VendorEmbeddingResponse(
            success=False,
            message=f"Error regenerating vendor embeddings: {str(e)}",
            vendor_count=0,
            embeddings_generated=False,
            faiss_index_built=False
        )


@app.get("/api/vendor/embeddings/status")
async def get_vendor_embeddings_status():
    """Check status of vendor embeddings and FAISS index"""
    try:
        vendor_service = VendorEmbeddingService()
        
        # Check if files exist
        embeddings_exist = os.path.exists(vendor_service.vendor_embeddings_file)
        faiss_index_exist = os.path.exists(vendor_service.vendor_index_file)
        
        # Try to load if they exist
        embeddings_ready = False
        faiss_ready = False
        vendor_count = 0
        
        if embeddings_exist:
            try:
                vendor_service.load_vendor_embeddings()
                embeddings_ready = True
                vendor_count = len(vendor_service.vendor_locations)
            except Exception as e:
                print(f"Error loading vendor embeddings: {e}")
        
        if faiss_index_exist:
            try:
                vendor_service.load_vendor_faiss_index()
                faiss_ready = True
            except Exception as e:
                print(f"Error loading vendor FAISS index: {e}")
        
        return {
            "embeddings_exist": embeddings_exist,
            "faiss_index_exist": faiss_index_exist,
            "embeddings_ready": embeddings_ready,
            "faiss_ready": faiss_ready,
            "vendor_count": vendor_count,
            "overall_ready": embeddings_ready and faiss_ready
        }
        
    except Exception as e:
        return {
            "embeddings_exist": False,
            "faiss_index_exist": False,
            "embeddings_ready": False,
            "faiss_ready": False,
            "vendor_count": 0,
            "overall_ready": False,
            "error": str(e)
        }

# ============================================================================
# CRON JOB SETUP
# ============================================================================

def run_vendor_embeddings_cron():
    """Function to run vendor embeddings cron job"""
    try:
        print("🕛 CRON JOB: Starting scheduled vendor embedding regeneration...")
        port = os.environ.get("PORT", "8000")
        base_url = f"http://localhost:{port}"
        response = requests.post(f"{base_url}/api/vendor/embeddings/cron")
        result = response.json()
        print(f"🕛 CRON JOB: {result['message']}")
    except Exception as e:
        print(f"🕛 CRON JOB ERROR: {e}")

def start_cron_scheduler():
    """Start the cron scheduler in a separate thread"""
    def run_scheduler():
        # Schedule vendor embeddings regeneration daily at 10:24 PM
        schedule.every().day.at("00:00").do(run_vendor_embeddings_cron)
        
        print("🕛 CRON SCHEDULER: Started - Vendor embeddings will regenerate daily at 12:00 AM")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    # Start scheduler in background thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    print("✅ CRON SCHEDULER: Background thread started")


# Start the FastAPI server
if __name__ == "__main__":
    
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get("PORT", 8000))
    
    # Start server - must listen on 0.0.0.0 for Render to detect the port
    uvicorn.run(app, host="0.0.0.0", port=port)