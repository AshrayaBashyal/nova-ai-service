from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import asyncio

from app.services.llm_service import GeminiService
from app.schemas.chat import (
    ChatStreamRequest, ChatFullRequest, 
    TokenCountRequest, TokenCountResponse, ChatFullResponse
)


router = APIRouter()
gemini = GeminiService()


@router.post("/v1/chat/stream")
async def chat_stream(req: ChatStreamRequest):
    """Stream chat from Gemini with optional token usage at the end."""
    return StreamingResponse(
        gemini.stream_chat_with_usage(
            req.messages, 
            system_prompt=req.system_prompt,
            include_usage=req.include_usage
        ),
        media_type="text/event-stream"
    )


@router.post("/v1/chat/full", response_model=ChatFullResponse)
async def chat_full(req: ChatFullRequest):
    """
    Get complete response (non-streaming) with token usage.
    """
    try:
        response = await gemini.chat_full(
            req.messages,
            system_prompt=req.system_prompt
        )
        return response  
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