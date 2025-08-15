"""
Embedding generation module for Vertex AI Vector Search.

This module provides functionality to convert text into vector embeddings
using Google's Vertex AI embedding models.
"""

import logging
from typing import List, Optional
import numpy as np

import vertexai
from vertexai.language_models import TextEmbeddingModel

from .config import config
from google.api_core import retry


logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    A class to generate embeddings from text using Vertex AI's high-level SDK.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.embedding_model
        self.batch_size = config.embedding_batch_size
        self.location = config.location.strip()
        self._initialize_vertex_ai()

        # Cargar el modelo de embeddings
        try:
            self.model = TextEmbeddingModel.from_pretrained(self.model_name)
            logger.info(f"Modelo de embeddings inicializado: {self.model_name}")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo de embeddings '{self.model_name}': {e}")
            raise

    def _initialize_vertex_ai(self) -> None:
        """Initialize Vertex AI client and set up the project."""
        try:
            vertexai.init(project=config.project_id, location=self.location)
            logger.info(
                f"Vertex AI inicializado para el proyecto: {config.project_id} en la región: {self.location}"
            )
        except Exception as e:
            logger.error(f"Error inicializando Vertex AI: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a single embedding from text.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or None")

        try:
            logger.debug(
                f"Generando embedding individual. Longitud del texto: {len(text)}"
            )
            embeddings = self.model.get_embeddings([text])
            if not embeddings:
                raise ValueError("No se recibieron embeddings del modelo")

            embedding_values = embeddings[0].values
            logger.debug(
                f"Embedding generado. Dimensión: {len(embedding_values)}"
            )
            return embedding_values
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    @retry.Retry(predicate=retry.if_exception_type(Exception))
    def generate_embeddings_batch(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("No valid texts found in the input list")

        batch_size = batch_size or self.batch_size
        all_embeddings: List[List[float]] = []

        logger.info(
            f"Procesando {len(valid_texts)} textos válidos de un total de {len(texts)}"
        )

        try:
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i : i + batch_size]
                if not batch:
                    continue

                logger.debug(
                    f"Solicitando embeddings para el batch {i//batch_size + 1}/"
                    f"{(len(valid_texts)-1)//batch_size + 1} con {len(batch)} textos"
                )
                embeddings = self.model.get_embeddings(batch)
                if not embeddings:
                    logger.warning(
                        f"No se recibieron embeddings para el batch que inicia en índice {i}"
                    )
                    continue

                batch_vectors = [e.values for e in embeddings]
                all_embeddings.extend(batch_vectors)

            logger.info(f"Se generaron embeddings para {len(all_embeddings)} textos")
            return all_embeddings
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate that an embedding vector is properly formatted."""
        if not embedding:
            return False

        expected_dim = config.get_embedding_dimension()
        if len(embedding) != expected_dim:
            logger.warning(
                f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}"
            )
            return False

        if any(not np.isfinite(val) for val in embedding):
            logger.warning("Embedding contains NaN or infinite values")
            return False

        return True
