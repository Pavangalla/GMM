from pydantic import BaseModel
from typing import Optional, List

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = []
    user_id: str
    subscription_tier: str  # "industry" | "individual" | "country" | "enterprise"
    licensed_industries: Optional[List[str]] = None
    licensed_geographies: Optional[List[str]] = None

class ChatResponse(BaseModel):
    response: str
    tool_calls_made: List[str] = []