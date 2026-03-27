import os
import asyncio
from typing import AsyncGenerator, Optional
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv


load_dotenv()

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
            model_name="gemini-2.5-flash",
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1000,
                top_p=0.95,
            )
        )
    

    async def chat_stream(
            self,
            messages: list,
            system_prompt: Optional[str] = None,
            include_usage: bool = True
        ) -> AsyncGenerator[str, None]:
            """Stream response tokens and yield token usage at the end."""
            loop = asyncio.get_event_loop()
            output_text = ""

            # Prepare messages for Gemini
            gemini_messages = []
            for msg in messages:
                role = "model" if msg.role == "assistant" else msg.role
                gemini_messages.append({"role": role, "parts": [msg.content]})

            if system_prompt:
                gemini_messages.insert(0, {"role": "user", "parts": [f"System: {system_prompt}"]})

            # Count input tokens
            input_text = " ".join([msg.content for msg in messages])
            try:
                input_tokens = await loop.run_in_executor(_executor, lambda: self.model.count_tokens(input_text).total_tokens)
            except:
                input_tokens = len(input_text) // 4  # rough fallback

            try:
                response = await loop.run_in_executor(
                    _executor,
                    lambda: self.model.generate_content(gemini_messages, stream=True)
                )

                for chunk in response:
                    if chunk.text:
                        output_text += chunk.text
                        yield chunk.text

            except Exception as e:
                yield f"\n[ERROR] {type(e).__name__}: {str(e)}"

            finally:
                if include_usage:
                    try:
                        output_tokens = await loop.run_in_executor(_executor, lambda: self.model.count_tokens(output_text).total_tokens)
                    except:
                        output_tokens = len(output_text) // 4
                    total_tokens = input_tokens + output_tokens
                    import json
                    summary = {
                        "type": "token_summary",
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    }
                    yield "\n" + json.dumps(summary)
    

    async def chat_full(
        self, 
        messages: list,
        system_prompt: Optional[str] = None
    ) -> dict:
        """
        Get full response (non-streaming) with token usage.
        """
        loop = asyncio.get_event_loop()
        
        try:
            gemini_messages = []
            for msg in messages:
                role = "model" if msg.role == "assistant" else msg.role
                gemini_messages.append({
                    "role": role,
                    "parts": [msg.content]
                })
            
            if system_prompt:
                gemini_messages.insert(0, {
                    "role": "user",
                    "parts": [f"System: {system_prompt}"]
                })
            
            # Build prompt text
            prompt_text = " ".join([msg.content for msg in messages])
            if system_prompt:
                prompt_text = system_prompt + " " + prompt_text
            
            # Count input tokens
            input_tokens = await self.count_tokens(prompt_text)
            
            # Generate response
            response = await loop.run_in_executor(
                _executor,
                lambda: self.model.generate_content(gemini_messages)
            )
            
            # Count output tokens
            output_tokens = await self.count_tokens(response.text)
            
            return {
                "content": response.text,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
            }
        
        except Exception as e:
            return {
                "content": f"[ERROR] {type(e).__name__}: {str(e)}",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }
            }
    
    
    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in !!TEST!! using Gemini's tokenizer.
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