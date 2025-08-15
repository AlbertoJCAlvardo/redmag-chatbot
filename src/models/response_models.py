"""
Response models for MentorIA Chatbot API.

Pydantic models for API response validation and serialization.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    """Model for chat interaction responses."""
    
    conversation_id: str = Field(..., description="Current conversation ID")
    response_type: str = Field(..., description="Response type to render: 'text', 'buttons', or 'content_cards'")
    data: Dict[str, Any] = Field(..., description="Data payload corresponding to response_type")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ContentResponse(BaseModel):
    """Model for content operation responses."""
    
    id: str = Field(..., description="Content unique identifier")
    title: str = Field(..., description="Content title")
    description: str = Field(..., description="Content description")
    content: str = Field(..., description="Main content text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Content tags")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class ContentListResponse(BaseModel):
    """Model for content list responses."""
    
    items: List[ContentResponse] = Field(..., description="List of content items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Page size")


class SuccessResponse(BaseModel):
    """Model for successful operation responses."""
    
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional response data")


class ErrorResponse(BaseModel):
    """Model for error responses."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow) 