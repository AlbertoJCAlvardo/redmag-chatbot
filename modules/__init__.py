"""
Vertex AI Vector Search Modules Package.

This package contains all the modules for Vertex AI Vector Search operations
including embeddings, vector search, and CMS integration.
"""

__version__ = "1.0.0"
__author__ = "RedMag Chatbot Team"

from .config import config
from .embeddings import EmbeddingGenerator
from .vector_search import VectorSearchManager
from .cms_integration import CMSIntegration, WordPressConnector

__all__ = [
    "config",
    "EmbeddingGenerator",
    "VectorSearchManager", 
    "CMSIntegration",
    "WordPressConnector"
]