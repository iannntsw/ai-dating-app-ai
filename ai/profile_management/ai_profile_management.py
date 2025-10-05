import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv(".env")

def init_ai():
    os.environ["LANGSMITH_TRACING"] = "false"
    
    # Get API key with proper error handling
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    # Set the API key in environment
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Use the faster init_chat_model approach with speed optimizations
    model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    return model

