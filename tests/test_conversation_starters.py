#!/usr/bin/env python3
"""
Test script for conversation starters endpoint
"""

import requests
import json

def test_conversation_starters():
    """Test the conversation starters endpoint"""
    
    # Test data
    test_data = {
        "user1": {
            "name": "John",
            "age": 25,
            "gender": "male",
            "interests": ["photography", "hiking", "coffee"],
            "bio": "Love outdoor adventures and capturing moments",
            "job": "Photographer",
            "education": "College",
            "location": "New York"
        },
        "user2": {
            "name": "Sarah",
            "age": 23,
            "gender": "female", 
            "interests": ["photography", "coffee", "travel"],
            "bio": "Coffee enthusiast and travel photographer",
            "job": "Designer",
            "education": "University",
            "location": "New York"
        },
        "sharedInterests": ["photography", "coffee"]
    }
    
    try:
        # Make request to AI service
        response = requests.post(
            "http://127.0.0.1:8000/ai/generate-conversation-starters",
            json=test_data,
            timeout=30
        )
        
        print("Status Code:", response.status_code)
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                starters = data.get("starters", [])
                print(f"\n✅ Success! Generated {len(starters)} conversation starters:")
                for i, starter in enumerate(starters, 1):
                    print(f"{i}. {starter.get('text', 'N/A')}")
                    print(f"   Type: {starter.get('type', 'N/A')}")
                    print(f"   Category: {starter.get('category', 'N/A')}")
                    print(f"   Confidence: {starter.get('confidence', 'N/A')}")
                    print()
            else:
                print("❌ AI service returned success=false")
                print("Error:", data.get("error", "Unknown error"))
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print("Response:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not connect to AI service")
        print("Make sure the AI service is running on http://127.0.0.1:8000")
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: Request took too long")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Testing Conversation Starters Endpoint")
    print("=" * 50)
    test_conversation_starters()
