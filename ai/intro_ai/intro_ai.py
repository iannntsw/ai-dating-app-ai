"""
AI Introduction Message Generator

This module generates personalized, engaging introduction messages for dating app users
based on different contexts (bio, interests, prompts, or general).
"""

from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict, List, Optional
import os

# Cache for loaded prompts
_PROMPTS_CACHE = None


def _load_prompts_from_file() -> Dict[str, str]:
    """Load all system prompts from intro_ai.md file."""
    global _PROMPTS_CACHE
    
    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE
    
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(current_dir, "intro_ai.md")
    
    # Read the prompts file
    with open(prompt_file, "r") as f:
        content = f.read()
    
    # Parse the prompts
    prompts = {}
    sections = content.split("===")
    
    for section in sections:
        if not section.strip():
            continue
        
        lines = section.strip().split("\n", 1)
        if len(lines) < 2:
            continue
        
        # Extract prompt type and content
        prompt_type = lines[0].replace("_PROMPT", "").lower()
        prompt_content = lines[1].strip()
        
        prompts[prompt_type] = prompt_content
    
    _PROMPTS_CACHE = prompts
    return prompts


def _load_system_prompt(context_type: str) -> str:
    """Load system prompt for the given context type from intro_ai.md."""
    prompts = _load_prompts_from_file()
    return prompts.get(context_type, prompts.get("general", ""))


def _build_human_prompt(context_type: str, name: str, **context_data) -> str:
    """
    Build the human prompt based on context type and data.
    
    Args:
        context_type: One of "bio", "interests", "prompt", or "general"
        name: The person's first name
        **context_data: Additional context data (bio, interests, question, answer, user_name, existing_message)
    
    Returns:
        Human prompt string
    """
    user_name = context_data.get("user_name", "")
    existing_message = context_data.get("existing_message", "")
    
    user_context = f"The user's name is {user_name}. " if user_name else ""
    existing_context = f"\n\nThe user has already started writing:\n\"{existing_message}\"\n\nImprove or complete this message while keeping their voice." if existing_message and existing_message.strip() else ""
    
    if context_type == "bio":
        bio = context_data.get("bio", "")
        return f"""{user_context}Generate an INTRODUCTION message (MAX 300 characters) for someone with this bio:

"{bio}"

CRITICAL: Help the user introduce THEMSELVES, not just comment. The user must share something about their own experience or interest that relates to this bio. Find common ground and end with a question. Write like texting - no "Hi {name}," greeting.{existing_context}"""

    elif context_type == "interests":
        interests = context_data.get("interests", [])
        interests_str = ", ".join(interests) if interests else "various hobbies"
        return f"""{user_context}Generate an INTRODUCTION message (MAX 300 characters) for someone interested in: {interests_str}

CRITICAL: Help the user introduce THEMSELVES. The user must share their own experience with this interest (even if beginner) or relate it to themselves. Find common ground and end with a question. Write like texting - no formal greeting.{existing_context}"""

    elif context_type == "prompt":
        question = context_data.get("question", "")
        answer = context_data.get("answer", "")
        return f"""{user_context}Generate an INTRODUCTION message (MAX 300 characters) for someone who answered:

Question: "{question}"
Their answer: "{answer}"

CRITICAL: Help the user introduce THEMSELVES. The user must share their own perspective or relate the answer to their experience. Show you relate to their answer through YOUR lens and end with a question. Write like texting - no formal greeting.{existing_context}"""

    else:  # general
        return f"""{user_context}Generate a casual text message (MAX 300 characters) as a friendly opening.

Write like you're texting a friend. Start directly - no formal greeting. Be warm and end with an open-ended question.{existing_context}"""


def generate_introduction(
    model,
    context_type: str,
    name: str,
    bio: Optional[str] = None,
    interests: Optional[List[str]] = None,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    user_name: Optional[str] = None,
    existing_message: Optional[str] = None
) -> str:
    # Validate context type
    valid_types = ["bio", "interests", "prompt", "general"]
    if context_type not in valid_types:
        raise ValueError(f"Invalid context_type: {context_type}. Must be one of {valid_types}")
    
    # Validate required data for each context type
    if context_type == "bio" and not bio:
        raise ValueError("Bio is required for context_type='bio'")
    if context_type == "interests" and not interests:
        raise ValueError("Interests are required for context_type='interests'")
    if context_type == "prompt" and (not question or not answer):
        raise ValueError("Both question and answer are required for context_type='prompt'")
    
    # Load system prompt for this context type
    system_prompt_text = _load_system_prompt(context_type)
    system_prompt = SystemMessage(content=system_prompt_text)
    
    # Build human prompt with context data
    human_prompt_text = _build_human_prompt(
        context_type=context_type,
        name=name,
        bio=bio,
        interests=interests,
        question=question,
        answer=answer,
        user_name=user_name,
        existing_message=existing_message
    )
    human_prompt = HumanMessage(content=human_prompt_text)
    
    # Get AI response
    response = model.invoke([system_prompt, human_prompt])
    
    # Return cleaned message
    return response.content.strip()


def generate_introduction_from_request(model, request_data: Dict) -> Dict[str, str]:

    context_type = request_data.get("type", "general")
    name = request_data.get("name", "there")
    bio = request_data.get("bio")
    interests = request_data.get("interests")
    question = request_data.get("question")
    answer = request_data.get("answer")
    user_name = request_data.get("user_name")
    existing_message = request_data.get("existing_message")
    
    try:
        message = generate_introduction(
            model=model,
            context_type=context_type,
            name=name,
            bio=bio,
            interests=interests,
            question=question,
            answer=answer,
            user_name=user_name,
            existing_message=existing_message
        )
        return {"message": message}
    except ValueError as e:
        # If validation fails, fall back to general message
        print(f"Warning: {str(e)}. Falling back to general message.")
        message = generate_introduction(
            model=model,
            context_type="general",
            name=name,
            user_name=user_name,
            existing_message=existing_message
        )
        return {"message": message}

