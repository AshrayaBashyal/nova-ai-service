import os
import asyncio
from typing import AsyncGenerator, Optional
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=5)

class GeminiService:
    """
    Async wrapper for Gemini API.
    Handles streaming natively with Gemini's response format.
    """
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not in .env")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1000,
                top_p=0.95,
            )
        )
    
    async def stream_chat(
        self, 
        messages: list,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Gemini one-by-one.
        
        Args:
            messages: List of {"role": "user"/"model", "content": "..."}.
                      Gemini uses "model" instead of "assistant".
            system_prompt: Optional system instruction.
        
        Yields:
            Text chunks as they arrive from Gemini.
        """
        loop = asyncio.get_event_loop()
        
        try:
            # Convert message format if needed
            gemini_messages = []
            for msg in messages:
                role = "model" if msg.get("role") == "assistant" else msg.get("role", "user")
                gemini_messages.append({
                    "role": role,
                    "parts": [msg.get("content", "")]
                })
            
            # System prompt goes in generation config or as first user message
            if system_prompt:
                gemini_messages.insert(0, {
                    "role": "user",
                    "parts": [f"System: {system_prompt}"]
                })
            
            # Run blocking Gemini call in thread pool
            response = await loop.run_in_executor(
                _executor,
                lambda: self.model.generate_content(
                    gemini_messages,
                    stream=True
                )
            )
            
            # Stream tokens from response
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        
        except Exception as e:
            yield f"\n[ERROR] {type(e).__name__}: {str(e)}"
    
    async def chat_full(
        self, 
        messages: list,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Get full response (non-streaming).
        
        Args:
            messages: List of {"role": "user"/"model", "content": "..."}.
            system_prompt: Optional system instruction.
        
        Returns:
            Complete text response.
        """
        loop = asyncio.get_event_loop()
        
        try:
            gemini_messages = []
            for msg in messages:
                role = "model" if msg.get("role") == "assistant" else msg.get("role", "user")
                gemini_messages.append({
                    "role": role,
                    "parts": [msg.get("content", "")]
                })
            
            if system_prompt:
                gemini_messages.insert(0, {
                    "role": "user",
                    "parts": [f"System: {system_prompt}"]
                })
            
            response = await loop.run_in_executor(
                _executor,
                lambda: self.model.generate_content(gemini_messages)
            )
            
            return response.text
        
        except Exception as e:
            return f"[ERROR] {type(e).__name__}: {str(e)}"
    
    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using Gemini's tokenizer.
        Free and useful for context management.
        """
        loop = asyncio.get_event_loop()
        
        try:
            response = await loop.run_in_executor(
                _executor,
                lambda: self.model.count_tokens(text)
            )
            return response.total_tokens
        except:
            # Fallback: rough estimate (1 token ≈ 4 chars)
            return len(text) // 4