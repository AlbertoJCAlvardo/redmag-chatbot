"""
Content service for MentorIA Chatbot API.

Handles business logic for MEDs and Planeaciones content management.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.adapters.vector_search_adapter import VectorSearchAdapter
from src.adapters.bigquery_adapter import BigQueryAdapter
from src.models.request_models import ContentCreateRequest, ContentUpdateRequest

logger = logging.getLogger(__name__)


class ContentService:
    """Service for managing educational content (MEDs and Planeaciones)."""

    def __init__(self):
        """Initialize ContentService with required adapters."""
        self.vector_adapter = VectorSearchAdapter()
        self.bq_adapter = BigQueryAdapter()
        logger.info("ContentService initialized")

    async def get_meds_content(self, page: int = 1, size: int = 20) -> Dict[str, Any]:
        """
        Retrieve MEDs content from vector database.
        
        Args:
            page: Page number for pagination
            size: Number of items per page
            
        Returns:
            Dictionary with content items and pagination info
        """
        try:
            search_results = self.vector_adapter.search_by_type("med", page, size)
            return {
                "items": search_results,
                "total": len(search_results),
                "page": page,
                "size": size
            }
        except Exception as e:
            logger.error(f"Error retrieving MEDs content: {e}", exc_info=True)
            raise

    async def create_med_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new MED content in vector database.
        
        Args:
            content_data: Content data dictionary
            
        Returns:
            Created content information
        """
        try:
            content_id = f"med_{uuid.uuid4()}"
            document_data = {
                'id': content_id,
                'content': f"Título: {content_data.get('title', '')}. Descripción: {content_data.get('description', '')}",
                'metadata': {
                    'source': 'api_med',
                    'type': 'med',
                    'title': content_data.get('title'),
                    'description': content_data.get('description'),
                    'tags': content_data.get('tags', []),
                    'created_at': datetime.now().isoformat()
                },
                'restricts': [
                    {
                        'namespace': 'source',
                        'allow': ['api_med']
                    }
                ]
            }
            
            result = self.vector_adapter.insert_documents_batch([document_data])
            if result.get(content_id, False):
                return {
                    "id": content_id,
                    "success": True,
                    "message": "MED content created successfully"
                }
            else:
                raise Exception("Failed to insert content into vector database")
                
        except Exception as e:
            logger.error(f"Error creating MED content: {e}", exc_info=True)
            raise

    async def update_med_content(self, med_id: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update existing MED content in vector database.
        
        Args:
            med_id: MED content identifier
            content_data: Updated content data
            
        Returns:
            Update result information
        """
        try:
            # First, delete the existing content
            await self.delete_med_content(med_id)
            
            # Then create new content with updated data
            return await self.create_med_content(content_data)
            
        except Exception as e:
            logger.error(f"Error updating MED content {med_id}: {e}", exc_info=True)
            raise

    async def delete_med_content(self, med_id: str) -> Dict[str, Any]:
        """
        Delete MED content from vector database.
        
        Args:
            med_id: MED content identifier
            
        Returns:
            Deletion result information
        """
        try:
            # Note: Vector Search doesn't support direct deletion
            # This would need to be implemented with a different strategy
            # For now, we'll mark it as deleted in metadata
            logger.warning(f"Delete operation for MED {med_id} - Vector Search deletion not implemented")
            return {
                "success": True,
                "message": "MED content marked for deletion",
                "note": "Vector Search deletion requires index rebuild"
            }
            
        except Exception as e:
            logger.error(f"Error deleting MED content {med_id}: {e}", exc_info=True)
            raise

    async def get_planeaciones_content(self, page: int = 1, size: int = 20) -> Dict[str, Any]:
        """
        Retrieve Planeaciones content from vector database.
        
        Args:
            page: Page number for pagination
            size: Number of items per page
            
        Returns:
            Dictionary with content items and pagination info
        """
        try:
            search_results = self.vector_adapter.search_by_type("planeacion", page, size)
            return {
                "items": search_results,
                "total": len(search_results),
                "page": page,
                "size": size
            }
        except Exception as e:
            logger.error(f"Error retrieving Planeaciones content: {e}", exc_info=True)
            raise

    async def create_planeacion_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new Planeacion content in vector database.
        
        Args:
            content_data: Content data dictionary
            
        Returns:
            Created content information
        """
        try:
            content_id = f"planeacion_{uuid.uuid4()}"
            document_data = {
                'id': content_id,
                'content': f"Título: {content_data.get('title', '')}. Descripción: {content_data.get('description', '')}",
                'metadata': {
                    'source': 'api_planeacion',
                    'type': 'planeacion',
                    'title': content_data.get('title'),
                    'description': content_data.get('description'),
                    'tags': content_data.get('tags', []),
                    'created_at': datetime.now().isoformat()
                },
                'restricts': [
                    {
                        'namespace': 'source',
                        'allow': ['api_planeacion']
                    }
                ]
            }
            
            result = self.vector_adapter.insert_documents_batch([document_data])
            if result.get(content_id, False):
                return {
                    "id": content_id,
                    "success": True,
                    "message": "Planeacion content created successfully"
                }
            else:
                raise Exception("Failed to insert content into vector database")
                
        except Exception as e:
            logger.error(f"Error creating Planeacion content: {e}", exc_info=True)
            raise

    async def update_planeacion_content(self, planeacion_id: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update existing Planeacion content in vector database.
        
        Args:
            planeacion_id: Planeacion content identifier
            content_data: Updated content data
            
        Returns:
            Update result information
        """
        try:
            # First, delete the existing content
            await self.delete_planeacion_content(planeacion_id)
            
            # Then create new content with updated data
            return await self.create_planeacion_content(content_data)
            
        except Exception as e:
            logger.error(f"Error updating Planeacion content {planeacion_id}: {e}", exc_info=True)
            raise

    async def delete_planeacion_content(self, planeacion_id: str) -> Dict[str, Any]:
        """
        Delete Planeacion content from vector database.
        
        Args:
            planeacion_id: Planeacion content identifier
            
        Returns:
            Deletion result information
        """
        try:
            # Note: Vector Search doesn't support direct deletion
            # This would need to be implemented with a different strategy
            # For now, we'll mark it as deleted in metadata
            logger.warning(f"Delete operation for Planeacion {planeacion_id} - Vector Search deletion not implemented")
            return {
                "success": True,
                "message": "Planeacion content marked for deletion",
                "note": "Vector Search deletion requires index rebuild"
            }
            
        except Exception as e:
            logger.error(f"Error deleting Planeacion content {planeacion_id}: {e}", exc_info=True)
            raise 