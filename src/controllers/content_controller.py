"""
Content controller for MentorIA Chatbot API.

Handles HTTP requests for content management (MEDs and Planeaciones).
"""

import logging
from typing import Dict, Any

from src.services.content_service import ContentService

logger = logging.getLogger(__name__)


class ContentController:
    """Controller for handling content-related HTTP requests."""

    def __init__(self, content_service: ContentService):
        """Initialize ContentController with ContentService dependency."""
        self.content_service = content_service
        logger.info("ContentController initialized")

    async def get_meds_content(self) -> Dict[str, Any]:
        """Get MEDs content from vector database."""
        try:
            return await self.content_service.get_meds_content()
        except Exception as e:
            logger.error(f"Error in content controller getting MEDs: {e}", exc_info=True)
            raise

    async def create_med_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new MED content in vector database."""
        try:
            return await self.content_service.create_med_content(content_data)
        except Exception as e:
            logger.error(f"Error in content controller creating MED: {e}", exc_info=True)
            raise

    async def update_med_content(self, med_id: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing MED content in vector database."""
        try:
            return await self.content_service.update_med_content(med_id, content_data)
        except Exception as e:
            logger.error(f"Error in content controller updating MED {med_id}: {e}", exc_info=True)
            raise

    async def delete_med_content(self, med_id: str) -> Dict[str, Any]:
        """Delete MED content from vector database."""
        try:
            return await self.content_service.delete_med_content(med_id)
        except Exception as e:
            logger.error(f"Error in content controller deleting MED {med_id}: {e}", exc_info=True)
            raise

    async def get_planeaciones_content(self) -> Dict[str, Any]:
        """Get Planeaciones content from vector database."""
        try:
            return await self.content_service.get_planeaciones_content()
        except Exception as e:
            logger.error(f"Error in content controller getting Planeaciones: {e}", exc_info=True)
            raise

    async def create_planeacion_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new Planeacion content in vector database."""
        try:
            return await self.content_service.create_planeacion_content(content_data)
        except Exception as e:
            logger.error(f"Error in content controller creating Planeacion: {e}", exc_info=True)
            raise

    async def update_planeacion_content(self, planeacion_id: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing Planeacion content in vector database."""
        try:
            return await self.content_service.update_planeacion_content(planeacion_id, content_data)
        except Exception as e:
            logger.error(f"Error in content controller updating Planeacion {planeacion_id}: {e}", exc_info=True)
            raise

    async def delete_planeacion_content(self, planeacion_id: str) -> Dict[str, Any]:
        """Delete Planeacion content from vector database."""
        try:
            return await self.content_service.delete_planeacion_content(planeacion_id)
        except Exception as e:
            logger.error(f"Error in content controller deleting Planeacion {planeacion_id}: {e}", exc_info=True)
            raise 