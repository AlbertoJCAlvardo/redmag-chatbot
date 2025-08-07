"""
Embedding generation module for Vertex AI Vector Search.

This module provides functionality to convert text into vector embeddings
using Google's Vertex AI embedding models, with support for long texts
by splitting into token batches.
"""

import logging
import re
from typing import List, Optional, Union
import numpy as np
from google.cloud import aiplatform
from google.api_core import retry

from .config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    A class to generate embeddings from text using Vertex AI.
    
    This class handles the conversion of text data into vector embeddings
    that can be stored and searched in Vertex AI Vector Search.
    Supports long texts by splitting into token batches.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            model_name (Optional[str]): The embedding model to use. 
                                      Defaults to config.embedding_model.
        """
        self.model_name = model_name or config.embedding_model
        self.max_tokens = config.embedding_max_tokens
        self.batch_size = config.embedding_batch_size
        self._initialize_vertex_ai()
        
    def _initialize_vertex_ai(self) -> None:
        """Initialize Vertex AI client and set up the project."""
        try:
            # Setup authentication
            config.setup_authentication()
            
            aiplatform.init(
                project=config.project_id,
                location=config.location
            )
            logger.info(f"Initialized Vertex AI for project: {config.project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split long text into chunks that fit within token limits.
        
        Args:
            text (str): The text to split.
            
        Returns:
            List[str]: List of text chunks.
        """
        if not text or not text.strip():
            return []
        
        # Simple token estimation (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens_per_char = 0.25
        max_chars = int(self.max_tokens / estimated_tokens_per_char)
        
        if len(text) <= max_chars:
            return [text]
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed the limit
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If chunks are still too long, split by words
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final_chunks.append(chunk)
            else:
                words = chunk.split()
                current_chunk_words = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 > max_chars and current_chunk_words:
                        final_chunks.append(" ".join(current_chunk_words))
                        current_chunk_words = [word]
                        current_length = len(word)
                    else:
                        current_chunk_words.append(word)
                        current_length += len(word) + 1
                
                if current_chunk_words:
                    final_chunks.append(" ".join(current_chunk_words))
        
        return final_chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a single embedding from text.
        
        Args:
            text (str): The text to convert to embedding.
            
        Returns:
            List[float]: The embedding vector.
            
        Raises:
            ValueError: If text is empty or None.
            Exception: If embedding generation fails.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or None")
        
        try:
            # Split text into chunks if it's too long
            text_chunks = self._split_text_into_chunks(text)
            
            if len(text_chunks) == 1:
                # Single chunk, generate embedding directly
                embeddings = aiplatform.TextEmbedding(
                    model_name=self.model_name
                )
                result = embeddings.get_embeddings([text_chunks[0]])
                embedding = result[0].values
                
                logger.debug(f"Generated embedding for text of length: {len(text)}")
                return embedding
            else:
                # Multiple chunks, generate embeddings for each and average
                logger.info(f"Text too long, splitting into {len(text_chunks)} chunks")
                chunk_embeddings = self.generate_embeddings_batch(text_chunks)
                
                # Average the embeddings
                if chunk_embeddings:
                    avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
                    logger.info(f"Generated averaged embedding from {len(text_chunks)} chunks")
                    return avg_embedding
                else:
                    raise ValueError("Failed to generate embeddings for any chunks")
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    @retry.Retry(predicate=retry.if_exception_type(Exception))
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts (List[str]): List of texts to convert to embeddings.
            batch_size (Optional[int]): Size of batches to process. 
                                      Defaults to config.embedding_batch_size.
            
        Returns:
            List[List[float]]: List of embedding vectors.
            
        Raises:
            ValueError: If texts list is empty or contains invalid items.
            Exception: If batch embedding generation fails.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        
        try:
            # Process texts in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Filter out empty texts
                valid_batch = [text for text in batch if text and text.strip()]
                
                if not valid_batch:
                    logger.warning(f"Batch {i//batch_size + 1} contains no valid texts")
                    continue
                
                embeddings = aiplatform.TextEmbedding(
                    model_name=self.model_name
                )
                
                result = embeddings.get_embeddings(valid_batch)
                batch_embeddings = [emb.values for emb in result]
                
                # Pad with None for empty texts in original batch
                for j, text in enumerate(batch):
                    if not text or not text.strip():
                        batch_embeddings.insert(j, None)
                
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    def generate_embeddings_for_long_text(self, text: str) -> List[List[float]]:
        """
        Generate embeddings for a long text by splitting into chunks.
        
        Args:
            text (str): The long text to convert to embeddings.
            
        Returns:
            List[List[float]]: List of embedding vectors for each chunk.
            
        Raises:
            ValueError: If text is empty or None.
            Exception: If embedding generation fails.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or None")
        
        try:
            # Split text into chunks
            text_chunks = self._split_text_into_chunks(text)
            
            if not text_chunks:
                raise ValueError("No valid text chunks generated")
            
            logger.info(f"Split text into {len(text_chunks)} chunks")
            
            # Generate embeddings for each chunk
            embeddings = self.generate_embeddings_batch(text_chunks)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for long text: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings generated by the model.
        
        Returns:
            int: The embedding dimension.
        """
        return config.get_embedding_dimension()
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate that an embedding vector is properly formatted.
        
        Args:
            embedding (List[float]): The embedding vector to validate.
            
        Returns:
            bool: True if embedding is valid, False otherwise.
        """
        if not embedding:
            return False
        
        expected_dim = self.get_embedding_dimension()
        
        if len(embedding) != expected_dim:
            logger.warning(f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}")
            return False
        
        # Check for NaN or infinite values
        if any(not np.isfinite(val) for val in embedding):
            logger.warning("Embedding contains NaN or infinite values")
            return False
        
        return True
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text (str): The text to estimate tokens for.
            
        Returns:
            int: Estimated number of tokens.
        """
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4 