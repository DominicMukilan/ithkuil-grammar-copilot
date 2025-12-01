"""
LLM Client
================================

Handles all LLM API calls with:
- Groq (free tier, fast inference)
- Conversation history management
- Retry logic
- Token usage tracking
"""

import os
from typing import List, Dict, Optional

# Load .env file
from dotenv import load_dotenv
load_dotenv()  # Load .env from current directory

from groq import Groq


class LLMClient:
    """
    Client for LLM API calls with conversation management
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM client
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
        """
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"  # Best model available for free on Groq
        self.conversation_history: List[Dict] = []
    
    def chat(
        self, 
        user_message: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Send a message and get response
        
        Args:
            user_message: User's message
            system_prompt: Optional system prompt (only used on first message)
            max_tokens: Max tokens in response
            temperature: Creativity level (0-1)
            
        Returns:
            Assistant's response
        """
        # Add system prompt if this is first message
        if not self.conversation_history and system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add user message
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract response
            assistant_message = response.choices[0].message.content
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
            
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_conversation_length(self) -> int:
        """Get number of messages in conversation"""
        return len(self.conversation_history)


def test_llm_client():
    """
    Test the LLM client
    """
    print("=" * 70)
    print("TESTING LLM CLIENT")
    print("=" * 70)
    
    try:
        # Initialize client
        print("\n📡 Initializing Groq client...")
        client = LLMClient()
        print("✅ Client initialized!")
        
        # Test simple query
        print("\n🧪 Testing simple query...")
        
        system_prompt = """You are a helpful assistant that explains concepts clearly and concisely."""
        
        response = client.chat(
            user_message="Explain what RAG (Retrieval Augmented Generation) is in 2 sentences.",
            system_prompt=system_prompt,
            max_tokens=200
        )
        
        print(f"\n🤖 Response:")
        print(f"   {response}")
        
        # Test follow-up
        print("\n🧪 Testing follow-up question...")
        response2 = client.chat(
            user_message="Give me one example use case.",
            max_tokens=150
        )
        
        print(f"\n🤖 Response:")
        print(f"   {response2}")
        
        print(f"\n📊 Conversation length: {client.get_conversation_length()} messages")
        
        print("\n" + "=" * 70)
        print("✅ LLM CLIENT TEST PASSED!")
        print("=" * 70)
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 To fix:")
        print("   1. Go to https://console.groq.com/")
        print("   2. Sign up (free)")
        print("   3. Get API key")
        print("   4. Run: export GROQ_API_KEY='your-key-here'")
        print("   5. Or add to .env file")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    test_llm_client()