"""
Client for Firecrawl API with automatic key rotation and retry logic.
"""
import json
import os
import logging
import time
import random
import re
import threading
from typing import List, Dict, Any, Optional, Callable
from urllib.parse import urlparse

try:
    from firecrawl import FirecrawlApp
except ImportError:
    logging.warning("FirecrawlApp is not installed. Mocking class.")
    class FirecrawlApp:
        def __init__(self, api_key: str):
            raise ImportError("FirecrawlApp is not installed. Please run 'pip install firecrawl-py'")
        def extract(self, *args, **kwargs):
            raise NotImplementedError("FirecrawlApp is not available.")

def setup_logging(lang: str = "es") -> logging.Logger:
    """Sets up the logger for the module."""
    logger_instance = logging.getLogger(__name__)
    if not logger_instance.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s (Línea: %(lineno)d)')
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)
        logger_instance.setLevel(logging.INFO)
    return logger_instance

logger = setup_logging()

class EnhancedFirecrawlClient:
    """
    Client for Firecrawl API with key rotation and retry logic.
    """
    
    def __init__(self, api_keys: List[str], log_language: str = "es"):
        """Initializes the EnhancedFirecrawlClient."""
        self.logger = setup_logging(log_language)
        self._lock = threading.Lock()
        self.current_key_index = 0
        self.api_keys = api_keys
        
        if not self.api_keys:
            raise ValueError("API keys list cannot be empty.")
            
        self.firecrawl_app = FirecrawlApp(api_key=self.api_keys[0])

    def _validate_url(self, url: str) -> bool:
        """Validates if a URL is properly formatted."""
        if not isinstance(url, str):
            return False
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https') and bool(parsed.netloc)

    def _rotate_api_key(self) -> bool:
        """Rotates to the next available API key."""
        with self._lock:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            self.firecrawl_app = FirecrawlApp(api_key=self.api_keys[self.current_key_index])
            self.logger.info(f"Rotated API key to index {self.current_key_index}.")
            return True

    def _execute_with_retry(self, extraction_func: Callable, *args, **kwargs) -> Any:
        """Executes a function with retry and key rotation."""
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return extraction_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if "rate limit" in str(e).lower() or "insufficient credits" in str(e).lower():
                    if not self._rotate_api_key():
                        break
                time.sleep(1.5 ** attempt)
        
        raise Exception(f"Failed after {max_retries} attempts: {last_error}")

    def extract_url(self, url: str) -> Optional[str]:
        """
        Extracts clean content from a single URL using Firecrawl's extract method.
        """
        try:
            if not self._validate_url(url):
                self.logger.error(f"Invalid URL provided for extraction: {url}")
                return None
            
            # --- CORRECCIÓN AQUÍ ---
            # Se cambió 'extract_url' por el método correcto 'extract'
            response = self._execute_with_retry(
                self.firecrawl_app.extract,
                urls=[url],
                prompt = "extract the main content of the page in spanish"
            )
            
            # La respuesta de .extract() contiene la clave 'markdown' o 'content'
            if response:
                
                self.logger.info(f"Extraction successful for: {url}")
                self.logger.info(f"Extraction successful for: {response}")
                return response.data
            else:
                self.logger.warning(f"Could not retrieve content from URL: {url}. Response: {response}")
                return None
        except Exception as e:
            self.logger.error(f"Error during extraction of URL {url}: {e}")
            return None
