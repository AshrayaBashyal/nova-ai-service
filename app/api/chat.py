from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest, ChatFullResponse, TokenCountRequest, TokenCountResponse
from app.services.gemini_service import GeminiService

# Create the router instance
router = APIRouter()

# Dependency Injection: This ensures we have a service instance ready for each request.
def get_gemini_service():
    return GeminiService()

@router.post("/v1/chat/stream")
async def chat_stream(
    req: ChatRequest, 
    service: GeminiService = Depends(get_gemini_service)
):
    """
    SERVER-SENT EVENTS (SSE) ENDPOINT
    Streams Gemini's response chunk-by-chunk.
    """
    try:
        # We return a StreamingResponse which FastAPI handles efficiently.
        # media_type="text/event-stream" tells the browser to expect a stream.
        return StreamingResponse(
            service.chat_stream(
                messages=req.messages,
                system_prompt=req.system_prompt,
                temperature=req.temperature
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        # Standard FastAPI error handling
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

@router.post("/v1/tokens/count", response_model=TokenCountResponse)
async def count_tokens(
    req: TokenCountRequest, 
    service: GeminiService = Depends(get_gemini_service)
):
    """
    UTILITY ENDPOINT
    Allows the frontend to check if a long prompt will fit in the context window.
    """
    try:
        count = await service.count_tokens(req.text)
        return TokenCountResponse(tokens=count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))