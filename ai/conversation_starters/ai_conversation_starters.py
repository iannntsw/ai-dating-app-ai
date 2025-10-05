"""
AI Conversation Starters Module

This module generates personalized conversation starters for dating app users
based on their profiles, shared interests, and compatibility factors.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from ai.profile_management.ai_profile_management import init_ai
from typing import List, Dict, Any
import json

# Simple cache for faster responses
_conversation_cache = {}

def generate_conversation_starters(user1: Dict[str, Any], user2: Dict[str, Any], shared_interests: List[str], refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Generate personalized conversation starters for two users based on their profiles.
    
    Args:
        user1: Profile data for the current user
        user2: Profile data for the other user  
        shared_interests: List of shared interests between users
        
    Returns:
        List of conversation starter objects with text, type, category, and confidence
    """
    
    print("Debug: Starting conversation starter generation")
    print(f"Debug: user1 = {user1}")
    print(f"Debug: user2 = {user2}")
    print(f"Debug: shared_interests = {shared_interests}")
    
    # Check cache first for speed (unless refresh requested)
    cache_key = f"{user1.get('name', '')}_{user2.get('name', '')}_{','.join(shared_interests)}"
    if not refresh and cache_key in _conversation_cache:
        print("Debug: Using cached result for speed")
        return _conversation_cache[cache_key]
    
    try:
        # Initialize AI model
        print("Debug: Initializing AI model...")
        model = init_ai()
        print("Debug: AI model initialized successfully")
    except Exception as e:
        print(f"Debug: Error initializing AI model: {e}")
        return get_fallback_starters(user1, user2, shared_interests)
    
    # Dating-app optimized system prompt: quality, style, and strict JSON output
    system_prompt = SystemMessage(content=
        """
You are an expert dating coach crafting first-message conversation starters for a modern dating app.

GOALS
- Make it easy for USER 1 to message USER 2.
- Be specific to profile cues and any shared interests.
- Invite a reply with one clear, low-pressure question.

TONE
- Warm, playful, respectful, confidence without being try-hard.
- Natural, human, no clichés, no pickup lines, no emojis.

CONTENT RULES
- 12–22 words each. One sentence. End with a question mark.
- Vary structure across outputs (not all the same template).
- Use concrete details (interests, job, education, location) only if provided.
- If no shared interests, use observational/light curiosity.
- Avoid flattery about looks, avoid sensitive topics (politics, religion, income), avoid assumptions.

OUTPUT
- Return 3–5 items as a JSON array only (no prose, no backticks).
- Each item must be: {
  "text": string,
  "type": "interest" | "photo" | "recommendation" | "casual" | "personal",
  "category": short topic like "coffee", "hiking", or "general",
  "confidence": number between 0.6 and 0.95
}
"""
    )

    # Create human prompt with user data
    # Safely handle None values
    user1_interests = user1.get('interests', []) or []
    user2_interests = user2.get('interests', []) or []
    shared_interests_str = ', '.join(shared_interests) if shared_interests else 'None'
    
    human_prompt_content = f"Users: {user1.get('name', 'A')} ({user1.get('job', '')}) -> {user2.get('name', 'B')} ({user2.get('job', '')}). Shared: {shared_interests_str}. Generate 3 starters."

    human_prompt = HumanMessage(content=human_prompt_content)
    
    try:
        # Get AI response
        print("Debug: Calling AI model...")
        response = model.invoke([system_prompt, human_prompt])
        print("Debug: AI model responded successfully")
        print(f"Debug: AI response: {response.content[:200]}...")
        
        # Parse the JSON response
        try:
            # Try to extract JSON from the response
            response_text = response.content.strip()
            
            # Look for JSON array in the response
            if '[' in response_text and ']' in response_text:
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                json_text = response_text[start_idx:end_idx]
                
                starters = json.loads(json_text)
                
                # Validate and clean the response
                valid_starters = []
                for starter in starters:
                    if isinstance(starter, dict) and 'text' in starter:
                        valid_starters.append({
                            'text': starter.get('text', '').strip(),
                            'type': starter.get('type', 'casual'),
                            'category': starter.get('category', 'general'),
                            'confidence': float(starter.get('confidence', 0.8))
                        })
                
                # Cache the result for future speed
                _conversation_cache[cache_key] = valid_starters[:5]
                return valid_starters[:5]  # Return max 5 starters
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing AI response as JSON: {e}")
            
    except Exception as e:
        print(f"Error generating conversation starters: {e}")
    
    # Fallback: return basic conversation starters
    return get_fallback_starters(user1, user2, shared_interests)

def get_fallback_starters(user1: Dict[str, Any], user2: Dict[str, Any], shared_interests: List[str]) -> List[Dict[str, Any]]:
    """
    Generate fallback conversation starters when AI generation fails.
    """
    starters = []
    
    # Add shared interest based starters
    if shared_interests:
        for interest in shared_interests[:2]:  # Max 2 interest-based
            starters.append({
                'text': f"I see we both love {interest}! What got you into it?",
                'type': 'interest',
                'category': interest.lower(),
                'confidence': 0.9
            })
    
    # Add general starters
    general_starters = [
        {
            'text': "Hey! Your photos look amazing! Where was that photo taken?",
            'type': 'photo',
            'category': 'general',
            'confidence': 0.8
        },
        {
            'text': "Coffee or tea? I'm always curious about people's morning rituals ☕",
            'type': 'casual',
            'category': 'lifestyle',
            'confidence': 0.7
        },
        {
            'text': "What's the best part of your day? I love hearing about people's routines!",
            'type': 'personal',
            'category': 'daily',
            'confidence': 0.8
        }
    ]
    
    starters.extend(general_starters)
    return starters[:5]  # Return max 5
