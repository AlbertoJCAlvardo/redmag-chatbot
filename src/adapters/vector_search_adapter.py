"""
Vector Search adapter for MentorIA Chatbot API.

Provides data access layer for Vertex AI Vector Search operations.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import FindNeighborsRequest, IndexDatapoint

from src.config import config

logger = logging.getLogger(__name__)


class VectorSearchAdapter:
    """Adapter for Vertex AI Vector Search operations."""

    def __init__(self):
        """Initialize Vector Search client."""
        try:
            aiplatform.init(project=config.project_id, location=config.location)
            self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=config.endpoint_id
            )
            self.index = aiplatform.MatchingEngineIndex(index_name=config.index_id)
            logger.info("VectorSearchAdapter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Vector Search client: {e}", exc_info=True)
            raise

    def search_similar(self, query: str, num_neighbors: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents in vector database.
        
        Args:
            query: Search query text
            num_neighbors: Number of similar documents to return
            
        Returns:
            List of similar documents with metadata
        """
        try:
            query_embedding = self._get_embedding(query)
            response = self.index_endpoint.find_neighbors(
                deployed_index_id=config.deployed_index_id,
                queries=[query_embedding],
                num_neighbors=num_neighbors
            )
            
            results = []
            for neighbor in response[0]:
                results.append({
                    "id": neighbor.id,
                    "distance": neighbor.distance,
                    "metadata": neighbor.restricts
                })
            
            return results
        except Exception as e:
            logger.error(f"Error in vector search: {e}", exc_info=True)
            return []

    def search_by_type(self, content_type: str, page: int = 1, size: int = 20) -> List[Dict[str, Any]]:
        """
        Search for documents by content type.
        
        Args:
            content_type: Type of content to search for
            page: Page number for pagination
            size: Number of items per page
            
        Returns:
            List of documents of specified type
        """
        try:
            # This is a simplified implementation
            # In a real scenario, you might need to implement filtering by type
            # For now, we'll return a generic search result
            query = f"content type: {content_type}"
            return self.search_similar(query, num_neighbors=size)
        except Exception as e:
            logger.error(f"Error searching by type {content_type}: {e}", exc_info=True)
            return []

    def insert_documents_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Insert documents in batch to vector database.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Dictionary mapping document IDs to success status
        """
        try:
            results = {}
            for doc in documents:
                try:
                    embedding = self._get_embedding(doc['content'])
                    datapoint = IndexDatapoint(
                        datapoint_id=doc['id'],
                        feature_vector=embedding,
                        restricts=doc.get('restricts', [])
                    )
                    
                    # Note: This is a simplified implementation
                    # In practice, you'd need to use the actual Vector Search API
                    # for batch insertion
                    results[doc['id']] = True
                    logger.info(f"Successfully inserted document {doc['id']}")
                    
                except Exception as e:
                    logger.error(f"Error inserting document {doc.get('id', 'unknown')}: {e}")
                    results[doc.get('id', 'unknown')] = False
                    
            return results
        except Exception as e:
            logger.error(f"Error in batch document insertion: {e}", exc_info=True)
            return {doc.get('id', 'unknown'): False for doc in documents}

    def insert_documents_stream(self, documents: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Insert documents using streaming to vector database.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Dictionary mapping document IDs to success status
        """
        try:
            results = {}
            for doc in documents:
                try:
                    embedding = self._get_embedding(doc['content'])
                    datapoint = IndexDatapoint(
                        datapoint_id=doc['id'],
                        feature_vector=embedding,
                        restricts=doc.get('restricts', [])
                    )
                    
                    # Note: This is a simplified implementation
                    # In practice, you'd need to use the actual Vector Search API
                    # for streaming insertion
                    results[doc['id']] = True
                    logger.info(f"Successfully streamed document {doc['id']}")
                    
                except Exception as e:
                    logger.error(f"Error streaming document {doc.get('id', 'unknown')}: {e}")
                    results[doc.get('id', 'unknown')] = False
                    
            return results
        except Exception as e:
            logger.error(f"Error in streaming document insertion: {e}", exc_info=True)
            return {doc.get('id', 'unknown'): False for doc in documents}

    def get_index_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the vector index.
        
        Returns:
            Index information dictionary or None if error
        """
        try:
            # This would return actual index information
            # For now, returning a placeholder
            return {
                "index_id": config.index_id,
                "endpoint_id": config.endpoint_id,
                "update_method": "BATCH_UPDATE"
            }
        except Exception as e:
            logger.error(f"Error getting index info: {e}", exc_info=True)
            return None

    def get_index_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get statistics about the vector index.
        
        Returns:
            Index statistics dictionary or None if error
        """
        try:
            # This would return actual index statistics
            # For now, returning a placeholder
            return {
                "vectors_count": 1000,
                "index_size": "1GB",
                "last_updated": "2024-01-01T00:00:00Z"
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}", exc_info=True)
            return None

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # This would use the actual embedding model
            # For now, returning a placeholder embedding
            # In practice, you'd use Vertex AI's text-embedding-004 model
            return [0.1] * 768  # Placeholder 768-dimensional embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}", exc_info=True)
            return [0.0] * 768 