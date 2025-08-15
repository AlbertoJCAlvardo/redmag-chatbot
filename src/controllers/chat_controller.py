"""
Chat controller for MentorIA Chatbot API.

Handles HTTP requests for chat interactions.
"""

import logging
from typing import Dict, Any, Optional, List

from src.services.chat_service import ChatService
from src.models.request_models import UserDataInput

logger = logging.getLogger(__name__)


class ChatController:
    """Controller for handling chat-related HTTP requests."""

    def __init__(self, chat_service: ChatService):
        """Initialize ChatController with ChatService dependency."""
        self.chat_service = chat_service
        logger.info("ChatController initialized")

    async def handle_interaction(
        self, 
        user_id: str, 
        conversation_id: str, 
        message: Optional[str], 
        user_data_input: Optional[List[UserDataInput]]
    ) -> Dict[str, Any]:
        """
        Handle chat interaction request.
        
        Args:
            user_id: Unique user identifier
            conversation_id: Conversation session ID
            message: User text message
            user_data_input: Structured data from user buttons
            
        Returns:
            Response data with type and content
        """
        try:
            response = await self.chat_service.handle_interaction(
                user_id=user_id,
                conversation_id=conversation_id,
                message=message,
                user_data_input=user_data_input
            )
            
            # Si el conversation_id cambió, actualizar la respuesta
            if "conversation_id" in response:
                new_conversation_id = response["conversation_id"]
                if new_conversation_id != conversation_id:
                    conversation_id = new_conversation_id
                    response["new_conversation"] = True
                    logger.info(f"Nueva conversación iniciada: {conversation_id}")
                else:
                    response["new_conversation"] = False
            
            return response
        except Exception as e:
            logger.error(f"Error in chat controller: {e}", exc_info=True)
            raise 