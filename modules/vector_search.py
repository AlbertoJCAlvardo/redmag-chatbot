"""
Vector Search operations module for Vertex AI.

This module provides functionality to manage documents in Vertex AI Vector Search
including insertion, deletion, updates, and search operations.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
from google.cloud.aiplatform_v1 import (
    IndexServiceClient,
    IndexEndpointServiceClient,
    UpsertDatapointsRequest,
    DeleteDatapointsRequest,
    FindNeighborsRequest,
    ReadIndexDatapointsRequest
)
from google.cloud.aiplatform_v1.types import (
    IndexDatapoint,
    Datapoint,
    FindNeighborsResponse
)
from google.api_core import retry

from config import config
from embeddings import EmbeddingGenerator

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)


class VectorSearchManager:
    """
    A class to manage Vector Search operations in Vertex AI.
    
    This class handles document operations including insertion, deletion,
    updates, and search functionality for the vector database.
    """
    
    def __init__(self):
        """Initialize the VectorSearchManager with necessary clients."""
        self._initialize_clients()
        self.embedding_generator = EmbeddingGenerator()
        
    def _initialize_clients(self) -> None:
        """Initialize Google Cloud clients for vector search operations."""
        try:
            config.validate()
            config.setup_authentication()
            
            self.index_service_client = IndexServiceClient()
            self.index_endpoint_service_client = IndexEndpointServiceClient()
            
            self.index_path = config.get_index_path()
            self.endpoint_path = config.get_endpoint_path()
            
            logger.info(f"Initialized Vector Search Manager for index: {self.index_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vector Search Manager: {e}")
            raise
    
    def insert_document(
        self, 
        document_id: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Insert a single document into the vector index.
        
        Args:
            document_id (str): Unique identifier for the document.
            content (str): Text content to be embedded and stored.
            metadata (Optional[Dict[str, Any]]): Additional metadata for the document.
            
        Returns:
            bool: True if insertion was successful, False otherwise.
            
        Raises:
            ValueError: If document_id or content is invalid.
            Exception: If insertion fails.
        """
        if not document_id or not document_id.strip():
            raise ValueError("Document ID cannot be empty")
        
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        
        try:
            # Generate embedding for the content
            embedding = self.embedding_generator.generate_embedding(content)
            
            if not self.embedding_generator.validate_embedding(embedding):
                raise ValueError("Generated embedding is invalid")
            
            # Prepare metadata
            doc_metadata = metadata or {}
            doc_metadata.update({
                "content": content,
                "created_at": datetime.utcnow().isoformat(),
                "document_id": document_id
            })
            
            # Create datapoint
            datapoint = IndexDatapoint(
                datapoint_id=document_id,
                feature_vector=embedding,
                restricts=[],
                crowding_tag="",
                sparse_embedding=None
            )
            
            # Prepare request
            request = UpsertDatapointsRequest(
                index=self.index_path,
                datapoints=[datapoint]
            )
            
            # Execute upsert
            operation = self.index_service_client.upsert_datapoints(request)
            operation.result()  # Wait for completion
            
            logger.info(f"Successfully inserted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert document {document_id}: {e}")
            raise
    
    @retry.Retry(predicate=retry.if_exception_type(Exception))
    def insert_documents_batch(
        self, 
        documents: List[Dict[str, Any]], 
        batch_size: Optional[int] = None
    ) -> Dict[str, bool]:
        """
        Insert multiple documents into the vector index in batches.
        
        Args:
            documents (List[Dict[str, Any]]): List of documents with keys:
                - 'id': Document ID
                - 'content': Text content
                - 'metadata': Optional metadata
            batch_size (Optional[int]): Size of batches to process.
            
        Returns:
            Dict[str, bool]: Dictionary mapping document IDs to success status.
            
        Raises:
            ValueError: If documents list is empty or malformed.
            Exception: If batch insertion fails.
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        batch_size = batch_size or config.batch_size
        results = {}
        
        try:
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Extract content for embedding generation
                contents = [doc.get('content', '') for doc in batch]
                valid_indices = [j for j, content in enumerate(contents) 
                               if content and content.strip()]
                
                if not valid_indices:
                    logger.warning(f"Batch {i//batch_size + 1} contains no valid content")
                    continue
                
                # Generate embeddings for valid content
                valid_contents = [contents[j] for j in valid_indices]
                embeddings = self.embedding_generator.generate_embeddings_batch(valid_contents)
                
                # Create datapoints
                datapoints = []
                for j, idx in enumerate(valid_indices):
                    doc = batch[idx]
                    doc_id = doc.get('id')
                    content = doc.get('content', '')
                    metadata = doc.get('metadata', {})
                    
                    if not doc_id:
                        logger.warning(f"Document at index {idx} has no ID, skipping")
                        continue
                    
                    embedding = embeddings[j]
                    if not self.embedding_generator.validate_embedding(embedding):
                        logger.warning(f"Invalid embedding for document {doc_id}, skipping")
                        continue
                    
                    # Prepare metadata
                    doc_metadata = metadata.copy()
                    doc_metadata.update({
                        "content": content,
                        "created_at": datetime.utcnow().isoformat(),
                        "document_id": doc_id
                    })
                    
                    datapoint = IndexDatapoint(
                        datapoint_id=doc_id,
                        feature_vector=embedding,
                        restricts=[],
                        crowding_tag="",
                        sparse_embedding=None
                    )
                    datapoints.append(datapoint)
                
                if datapoints:
                    # Prepare request
                    request = UpsertDatapointsRequest(
                        index=self.index_path,
                        datapoints=datapoints
                    )
                    
                    # Execute upsert
                    operation = self.index_service_client.upsert_datapoints(request)
                    operation.result()  # Wait for completion
                    
                    # Mark successful insertions
                    for datapoint in datapoints:
                        results[datapoint.datapoint_id] = True
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            logger.info(f"Successfully inserted {len([r for r in results.values() if r])} documents")
            return results
            
        except Exception as e:
            logger.error(f"Failed to insert documents batch: {e}")
            raise
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a single document from the vector index.
        
        Args:
            document_id (str): ID of the document to delete.
            
        Returns:
            bool: True if deletion was successful, False otherwise.
            
        Raises:
            ValueError: If document_id is invalid.
            Exception: If deletion fails.
        """
        if not document_id or not document_id.strip():
            raise ValueError("Document ID cannot be empty")
        
        try:
            request = DeleteDatapointsRequest(
                index=self.index_path,
                datapoint_ids=[document_id]
            )
            
            operation = self.index_service_client.delete_datapoints(request)
            operation.result()  # Wait for completion
            
            logger.info(f"Successfully deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise
    
    def delete_documents_batch(self, document_ids: List[str]) -> Dict[str, bool]:
        """
        Delete multiple documents from the vector index.
        
        Args:
            document_ids (List[str]): List of document IDs to delete.
            
        Returns:
            Dict[str, bool]: Dictionary mapping document IDs to success status.
            
        Raises:
            ValueError: If document_ids list is empty.
            Exception: If batch deletion fails.
        """
        if not document_ids:
            raise ValueError("Document IDs list cannot be empty")
        
        results = {}
        
        try:
            # Filter out empty IDs
            valid_ids = [doc_id for doc_id in document_ids if doc_id and doc_id.strip()]
            
            if not valid_ids:
                logger.warning("No valid document IDs provided for deletion")
                return results
            
            request = DeleteDatapointsRequest(
                index=self.index_path,
                datapoint_ids=valid_ids
            )
            
            operation = self.index_service_client.delete_datapoints(request)
            operation.result()  # Wait for completion
            
            # Mark successful deletions
            for doc_id in valid_ids:
                results[doc_id] = True
            
            logger.info(f"Successfully deleted {len(valid_ids)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Failed to delete documents batch: {e}")
            raise
    
    def update_document(
        self, 
        document_id: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing document in the vector index.
        
        Args:
            document_id (str): ID of the document to update.
            content (str): New text content for the document.
            metadata (Optional[Dict[str, Any]]): Updated metadata.
            
        Returns:
            bool: True if update was successful, False otherwise.
            
        Raises:
            ValueError: If document_id or content is invalid.
            Exception: If update fails.
        """
        if not document_id or not document_id.strip():
            raise ValueError("Document ID cannot be empty")
        
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        
        try:
            # Generate new embedding
            embedding = self.embedding_generator.generate_embedding(content)
            
            if not self.embedding_generator.validate_embedding(embedding):
                raise ValueError("Generated embedding is invalid")
            
            # Prepare metadata
            doc_metadata = metadata or {}
            doc_metadata.update({
                "content": content,
                "updated_at": datetime.utcnow().isoformat(),
                "document_id": document_id
            })
            
            # Create updated datapoint
            datapoint = IndexDatapoint(
                datapoint_id=document_id,
                feature_vector=embedding,
                restricts=[],
                crowding_tag="",
                sparse_embedding=None
            )
            
            # Upsert will update if exists, insert if not
            request = UpsertDatapointsRequest(
                index=self.index_path,
                datapoints=[datapoint]
            )
            
            operation = self.index_service_client.upsert_datapoints(request)
            operation.result()  # Wait for completion
            
            logger.info(f"Successfully updated document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            raise
    
    def search_similar(
        self, 
        query: str, 
        num_neighbors: int = 10,
        filter_expression: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using a query string.
        
        Args:
            query (str): The search query text.
            num_neighbors (int): Number of similar documents to return.
            filter_expression (Optional[str]): Filter expression for the search.
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with scores.
            
        Raises:
            ValueError: If query is invalid or endpoint not configured.
            Exception: If search fails.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not self.endpoint_path:
            raise ValueError("Vector endpoint not configured for search operations")
        
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            if not self.embedding_generator.validate_embedding(query_embedding):
                raise ValueError("Generated query embedding is invalid")
            
            # Prepare search request
            request = FindNeighborsRequest(
                index_endpoint=self.endpoint_path,
                deployed_index_id=config.index_id,
                queries=[query_embedding],
                num_neighbors=num_neighbors
            )
            
            if filter_expression:
                request.filter = filter_expression
            
            # Execute search
            response = self.index_endpoint_service_client.find_neighbors(request)
            
            # Process results
            results = []
            if response.nearest_neighbors:
                for neighbor in response.nearest_neighbors[0].neighbors:
                    result = {
                        "document_id": neighbor.datapoint.datapoint_id,
                        "distance": neighbor.distance,
                        "metadata": {}
                    }
                    
                    # Extract metadata if available
                    if hasattr(neighbor.datapoint, 'restricts'):
                        for restrict in neighbor.datapoint.restricts:
                            if hasattr(restrict, 'namespace') and hasattr(restrict, 'allow_list'):
                                result["metadata"][restrict.namespace] = restrict.allow_list
                    
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            raise
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document from the vector index.
        
        Args:
            document_id (str): ID of the document to retrieve.
            
        Returns:
            Optional[Dict[str, Any]]: Document data if found, None otherwise.
            
        Raises:
            ValueError: If document_id is invalid.
            Exception: If retrieval fails.
        """
        if not document_id or not document_id.strip():
            raise ValueError("Document ID cannot be empty")
        
        try:
            request = ReadIndexDatapointsRequest(
                index=self.index_path,
                datapoint_ids=[document_id]
            )
            
            response = self.index_service_client.read_index_datapoints(request)
            
            if response.datapoints:
                datapoint = response.datapoints[0]
                return {
                    "document_id": datapoint.datapoint_id,
                    "embedding": list(datapoint.feature_vector),
                    "metadata": {}
                }
            
            logger.info(f"Document {document_id} not found")
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve document {document_id}: {e}")
            raise 