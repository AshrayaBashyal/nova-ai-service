from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# --- Common Components ---

class Message(BaseModel):
    """
    Standardizes the conversation format.
    role: 'user' for the human, 'assistant' for the LLM.
    content: The actual text string.
    """
    role: Literal["user", "assistant"] = Field(
        ..., 
        description="The role of the message author. Must be 'user' or 'assistant'."
    )
    content: str = Field(
        ..., 
        min_length=1, 
        description="The text content of the message."
    )

class TokenUsage(BaseModel):
    """
    Captures the metadata provided by the Gemini API.
    Useful for monitoring costs and context limits.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

# --- Request Schemas ---

class ChatRequest(BaseModel):
    """
    The unified request body for both full and streaming chat.
    """
    messages: List[Message] = Field(
        ..., 
        description="A list of previous messages in the conversation."
    )
    system_prompt: Optional[str] = Field(
        None, 
        description="Optional instructions to guide the model's behavior."
    )
    temperature: Optional[float] = Field(
        0.7, 
        ge=0.0, 
        le=2.0, 
        description="Controls randomness. Higher is more creative."
    )

# --- Response Schemas ---

class ChatFullResponse(BaseModel):
    """
    The schema for a non-streaming, complete response.
    """
    content: str
    usage: TokenUsage
    model: str = "gemini-2.0-flash"

class TokenCountRequest(BaseModel):
    """Request for the standalone token counting endpoint."""
    text: str

class TokenCountResponse(BaseModel):
    """Response for the token counting endpoint."""
    tokens: int