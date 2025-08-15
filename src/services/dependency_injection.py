"""
Dependency injection for MentorIA Chatbot API.

Provides dependency injection functions for FastAPI endpoints.
"""

from functools import lru_cache
from src.services.chat_service import ChatService
from src.services.content_service import ContentService
from src.controllers.chat_controller import ChatController
from src.controllers.content_controller import ContentController


@lru_cache()
def get_chat_service() -> ChatService:
    """Get cached ChatService instance."""
    return ChatService()


@lru_cache()
def get_content_service() -> ContentService:
    """Get cached ContentService instance."""
    return ContentService()


@lru_cache()
def get_chat_controller() -> ChatController:
    """Get cached ChatController instance."""
    chat_service = get_chat_service()
    return ChatController(chat_service)


@lru_cache()
def get_content_controller() -> ContentController:
    """Get cached ContentController instance."""
    content_service = get_content_service()
    return ContentController(content_service) 