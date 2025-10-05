from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai.profile_management.ai_profile_management import init_ai
from ai.ai_lovabot.ai_lovabot import chat as lovabot_chat
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.ai_date_planner.ai_date_planner import AIDatePlanner
from ai.ai_date_planner.rule_engine import UserPreferences
from ai.conversation_starters.ai_conversation_starters import generate_conversation_starters

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
        print("✅ AI Date Planner initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize AI Date Planner: {e}")
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
    start_latitude: Optional[float] = None
    start_longitude: Optional[float] = None
    interests: List[str] = ["food", "culture", "nature"]
    budget_tier: str = "$$"
    date_type: str = "casual"
    exclusions: Optional[List[str]] = None  # Backend-only: What user does NOT want

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
