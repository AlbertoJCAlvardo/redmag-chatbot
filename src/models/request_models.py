"""
Request models for MentorIA Chatbot API.

Pydantic models for validating incoming request data.
"""

from typing import Any, Optional, List
from pydantic import BaseModel, Field


class UserDataInput(BaseModel):
    """Model for structured user data input from frontend buttons."""
    
    field: str = Field(..., description="User profile field to update, e.g. 'grado'")
    value: Any = Field(..., description="Value for the field, e.g. 'quinto'")


class ChatRequest(BaseModel):
    """Model for chat interaction requests."""
    
    user_id: str = Field(..., description="Unique user identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation session ID. Will be created if not provided")
    message: Optional[str] = Field(None, description="User text message")
    user_data: Optional[List[UserDataInput]] = Field(None, description="Structured data from user, typically from buttons")


class ContentCreateRequest(BaseModel):
    """Model for creating new content."""
    
    title: str = Field(..., description="Content title")
    description: str = Field(..., description="Content description")
    content: str = Field(..., description="Main content text")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")
    tags: Optional[List[str]] = Field(default_factory=list, description="Content tags")


class ContentUpdateRequest(BaseModel):
    """Model for updating existing content."""
    
    title: Optional[str] = Field(None, description="Content title")
    description: Optional[str] = Field(None, description="Content description")
    content: Optional[str] = Field(None, description="Main content text")
    metadata: Optional[dict] = Field(None, description="Additional metadata")
    tags: Optional[List[str]] = Field(None, description="Content tags") 