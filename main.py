from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai.profile_management.ai_profile_management import init_ai
from ai.ai_lovabot.ai_lovabot import chat as lovabot_chat
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
import subprocess
import json
import tempfile
import requests
from urllib.parse import urlparse
import base64

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.ai_date_planner.ai_date_planner import AIDatePlanner
from ai.ai_date_planner.rule_engine import UserPreferences

# --- AI Re-ranker imports ---
from ai.discover_profiles.models import Payload
from ai.discover_profiles.ranking import rank_recommendations

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
        print(f"Failed to initialize AI Date Planner: {e}")
        raise

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
    return "Hello, World! Connected to AI"
    

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

@app.post("/ai/lovabot")
def generate_lovabot_response(request: dict):
    # Extract messages from request
    messages = request.get("messages", [])
    
    if not messages:
        return {"answer": "No messages provided."}
    
    # Use lovabot chat function with RAG
    response = lovabot_chat(messages)
    return response

# Date Planner Models
class DatePlanRequest(BaseModel):
    start_time: str = "10:00"
    end_time: Optional[str] = None
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
