from pydantic import BaseModel, Field
from typing import List, Optional

class Message(BaseModel):
    """Single message in conversation."""
    role: str = Field(..., description="'user' or 'model'")
    content: str

class ChatStreamRequest(BaseModel):
    """Request for streaming endpoint."""
    messages: List[Message]
    system_prompt: Optional[str] = None
    include_usage: Optional[bool] = True

class ChatFullRequest(BaseModel):
    """Request for full response endpoint."""
    messages: List[Message]
    system_prompt: Optional[str] = None

class ChatFullResponseUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int

class ChatFullResponse(BaseModel):
    content: str
    usage: ChatFullResponseUsage    

class TokenCountRequest(BaseModel):
    """Request for token counting."""
    text: str

class TokenCountResponse(BaseModel):
    """Response with token count."""
    tokens: int
    text_length: int