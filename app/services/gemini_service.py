import json
from typing import AsyncGenerator, List, Optional
from google import genai
from google.genai import types
from app.core.config import settings
from app.schemas.chat import Message, TokenUsage

class GeminiService:
    """
    The Engine of the application. 
    Handles all communication with Google's Gemini 2.0 API.
    """
    
    def __init__(self):
        # The new SDK uses a unified Client object.
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_id = settings.GEMINI_MODEL

    async def chat_stream(
        self, 
        messages: List[Message], 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """
        Streams response chunks in real-time.
        Utilizes the 2026 async 'generate_content_stream' pattern.
        """
        
        # 1. Setup the Generation Config
        # We pass system_instruction NATIVELY here.
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
        )

        # 2. Convert our Pydantic messages to the SDK's 'Content' types
        contents = [
            types.Content(role=m.role, parts=[types.Part(text=m.content)])
            for m in messages
        ]

        try:
            # 3. Initialize the stream
            # The 'models.generate_content_stream' is the modern async way.
            stream = await self.client.aio.models.generate_content_stream(
                model=self.model_id,
                contents=contents,
                config=config
            )

            async for chunk in stream:
                if chunk.text:
                    yield json.dumps({"type": "chunk", "text": chunk.text}) + "\n"
                
                if chunk.usage_metadata:
                    yield json.dumps({
                        "type": "usage", 
                        "usage": {
                            "input_tokens": chunk.usage_metadata.prompt_token_count,
                            "output_tokens": chunk.usage_metadata.candidates_token_count,
                            "total_tokens": chunk.usage_metadata.total_token_count
                        }
                    }) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    async def count_tokens(self, text: str) -> int:
        """
        Standalone token counter. 
        Uses the high-speed 'count_tokens' endpoint.
        """
        response = await self.client.aio.models.count_tokens(
            model=self.model_id,
            contents=text
        )
        return response.total_tokens