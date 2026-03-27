from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import asyncio

from app.services.llm_service import GeminiService
from app.schemas.chat import (
    ChatStreamRequest, ChatFullRequest, 
    TokenCountRequest, TokenCountResponse
)

router = APIRouter()
gemini = GeminiService()

@router.post("/v1/chat/stream")
async def chat_stream(req: ChatStreamRequest):
    """
    Stream tokens from Gemini in real-time.
    
    Example:
        curl -N -X POST http://localhost:8000/api/v1/chat/stream \
          -H "Content-Type: application/json" \
          -d '{
            "messages": [{"role": "user", "content": "Hello"}],
            "system_prompt": "You are helpful."
          }'
    """
    async def token_generator():
        """Yield tokens as they arrive."""
        try:
            async for token in gemini.stream_chat(
                req.messages,
                system_prompt=req.system_prompt
            ):
                yield token
        except Exception as e:
            yield f"\n[ERROR] {str(e)}"
    
    return StreamingResponse(token_generator(), media_type="text/event-stream")

@router.post("/v1/chat/full")
async def chat_full(req: ChatFullRequest):
    """
    Get complete response (non-streaming).
    
    Use this when you need the full answer at once.
    """
    try:
        response = await gemini.chat_full(
            req.messages,
            system_prompt=req.system_prompt
        )
        return {"content": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/tokens/count")
async def count_tokens(req: TokenCountRequest) -> TokenCountResponse:
    """
    Count tokens in text.
    
    Useful for managing context length in Phase 2/3.
    Gemini's tokenizer is accurate and free to use.
    """
    try:
        count = await gemini.count_tokens(req.text)
        return TokenCountResponse(
            tokens=count,
            text_length=len(req.text)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    """Liveness check."""
    return {"status": "ok", "model": "gemini-2.0-flash"}