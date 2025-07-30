import json
import os
import logging
import time
import random
import re
import threading
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Attempt to import FirecrawlApp. Provide a mock if not installed.
try:
    from firecrawl import FirecrawlApp
except ImportError:
    logging.warning("FirecrawlApp is not installed. Some functionalities might be unavailable.")
    class FirecrawlApp:
        """Mock class for FirecrawlApp when the library is not installed."""
        def __init__(self, api_key: str):
            logging.error("FirecrawlApp is not available. Please install it with 'pip install firecrawl-py'")
            raise ImportError("FirecrawlApp is not installed. Please install it with 'pip install firecrawl-py'")
        def extract(self, *args, **kwargs):
            raise NotImplementedError("FirecrawlApp is not available.")
        def scrape(self, *args, **kwargs):
            raise NotImplementedError("FirecrawlApp is not available.")
        def search(self, *args, **kwargs):
            raise NotImplementedError("FirecrawlApp is not available.")


# Configure logging with a default English format
def setup_logging(lang: str = "en") -> logging.Logger:
    """
    Sets up the logger for the module, with configurable language.

    Args:
        lang (str): The language for log messages ('en' for English, 'es' for Spanish).
                    Defaults to 'en'.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger_instance = logging.getLogger(__name__)
    if not logger_instance.handlers: # Prevent adding multiple handlers if called multiple times
        handler = logging.StreamHandler()
        if lang == "es":
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s (Línea: %(lineno)d)'
            )
        else: # Default to English
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s (Line: %(lineno)d)'
            )
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)
        logger_instance.setLevel(logging.INFO) # Default level

    return logger_instance

logger = setup_logging() # Initialize logger with default English


class EnhancedFirecrawlClient:
    """
    Client for Firecrawl API with automatic key rotation and retry logic.

    This class provides methods to interact with the Firecrawl API for web scraping,
    structured data extraction, and web search, incorporating robust error handling,
    API key rotation, and retry mechanisms.
    """
    
    def __init__(self, api_keys: List[str], log_language: str = "en"):
        """
        Initializes the EnhancedFirecrawlClient.

        Args:
            api_keys (List[str]): A list of Firecrawl API keys to use for rotation.
            log_language (str): The language for log messages ('en' for English, 'es' for Spanish).
                                Defaults to 'en'.
        """
        self.logger = setup_logging(log_language)
        self._lock = threading.Lock()
        self.current_key_index = 0
        self.api_keys = api_keys
        
        if not self.api_keys:
            raise ValueError("API keys list cannot be empty.")
            
        self.firecrawl_app = FirecrawlApp(api_key=self.api_keys[0])
        self.failed_keys = set()
        self.key_usage_count = {i: 0 for i in range(len(api_keys))}
        self.advise_count = 0
        self.max_advices = 10 # Limit for initial logging advice messages

    def _clean_json_response(self, text: str) -> str:
        """
        Cleans and extracts JSON from response text.

        Args:
            text (str): The input text string, potentially containing JSON.

        Returns:
            str: A cleaned string containing a valid JSON object, or an empty object string.
        """
        if not text or not isinstance(text, str):
            return "{}"
        
        # Normalize single quotes to double quotes for valid JSON
        text = text.replace("'", '"')
        
        # Remove markdown code blocks (e.g., ```json ... ```)
        text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```\s*$', '', text, flags=re.IGNORECASE)
        
        # Extract the first valid JSON object
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx+1]
        
        # Remove problematic control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
        
    def _parse_firecrawl_response(self, response: Any) -> Dict[str, Any]:
        """
        Parses Firecrawl API response and extracts JSON data.

        Args:
            response (Any): The raw response object from Firecrawl.

        Returns:
            Dict[str, Any]: A dictionary representing the parsed JSON data,
                            or an error dictionary if parsing fails.
        """
        try:
            if self.advise_count < self.max_advices:
                self.logger.info(f"Response type: {type(response)}\nResponse: {response}")
                self.advise_count += 1
                
            if isinstance(response, dict) and 'content' in response:
                content = response['content']
                if isinstance(content, dict):
                    return content
                elif isinstance(content, str):
                    cleaned_text = self._clean_json_response(content)
                    return json.loads(cleaned_text)
            
            if hasattr(response, 'data') and response.data:
                data = response.data
                if isinstance(data, dict):
                    return data
                elif isinstance(data, str):
                    cleaned_text = self._clean_json_response(data)
                    return json.loads(cleaned_text)
            
            if isinstance(response, dict):
                return response
            
            response_str = str(response)
            if response_str.startswith('{') and response_str.endswith('}'):
                return json.loads(response_str)
            else:
                cleaned_text = self._clean_json_response(response_str)
                return json.loads(cleaned_text)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            return {
                "summary": "Error processing response",
                "extracted_fields": {},
                "error": f"JSON Error: {str(e)}"
            }
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            return {
                "summary": "Error processing response",
                "extracted_fields": {},
                "error": str(e)
            }

    def _validate_url(self, url: str) -> bool:
        """
        Validates if a URL is properly formatted.

        Args:
            url (str): The URL string to validate.

        Returns:
            bool: True if the URL is valid, False otherwise.
        """
        if isinstance(url, list):
            url = url[0] if url else ""
        if not isinstance(url, str):
            return False
        
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https') and bool(parsed.netloc)

    def _rotate_api_key(self, reason: str = "Rate limit or error") -> bool:
        """
        Rotates to the next available API key in the list.

        Args:
            reason (str): The reason for rotating the API key.

        Returns:
            bool: True if a new API key was successfully rotated to, False if all keys failed.
        """
        with self._lock:
            original_index = self.current_key_index
            attempts = 0
            
            while attempts < len(self.api_keys):
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                
                if self.current_key_index not in self.failed_keys:
                    try:
                        self.firecrawl_app = FirecrawlApp(api_key=self.api_keys[self.current_key_index])
                        self.logger.info(f"Rotated API key from index {original_index} to {self.current_key_index}. Reason: {reason}")
                        return True
                        
                    except Exception as e:
                        error_message = f"Failed to initialize API key {self.current_key_index}: {str(e)}"
                        self.logger.error(error_message)
                        self.failed_keys.add(self.current_key_index)

                attempts += 1
            
            self.logger.error("ALL API KEYS FAILED!")
            return False

    def _execute_with_retry(self, extraction_func: Callable, *args, **kwargs) -> Any:
        """
        Executes an extraction function with automatic retry and API key rotation.

        Args:
            extraction_func (Callable): The Firecrawl API method to execute (e.g., self.firecrawl_app.scrape).
            *args: Positional arguments to pass to the extraction function.
            **kwargs: Keyword arguments to pass to the extraction function.

        Returns:
            Any: The result from the successful execution of the extraction function.

        Raises:
            Exception: If the extraction fails after all retries and key rotations.
        """
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                with self._lock:
                    self.key_usage_count[self.current_key_index] += 1
                
                # Validate URLs before making the request if 'urls' param exists
                if 'urls' in kwargs:
                    urls = kwargs['urls']
                    if isinstance(urls, list):
                        valid_urls = [url for url in urls if self._validate_url(url)]
                        if not valid_urls:
                            raise ValueError("No valid URLs provided for extraction.")
                        kwargs['urls'] = valid_urls
                    else:
                        raise ValueError("The 'urls' parameter must be a list of strings.")
                
                result = extraction_func(*args, **kwargs)
                return result
                
            except Exception as e:
                last_error = e
                error_message = str(e).lower()
                error_type = "UNKNOWN"
                
                # Determine error type based on common Firecrawl errors
                if any(keyword in error_message for keyword in ['rate limit exceeded', 'status code 429']):
                    error_type = "RATE_LIMIT"
                elif any(keyword in error_message for keyword in ['insufficient credits', 'upgrade your plan']):
                    error_type = "INSUFFICIENT_CREDITS"
                elif 'timeout' in error_message or 'timed out' in error_message:
                    error_type = "TIMEOUT"
                elif any(keyword in error_message for keyword in ['invalid', 'url', 'all provided urls are invalid']):
                    error_type = "INVALID_URL"
                
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed with error type {error_type}: {str(e)}")
                
                # Handle different error types with specific retry logic
                if error_type == "RATE_LIMIT":
                    if not self._rotate_api_key(f"Rate limit: {str(e)}"):
                        break
                    time.sleep(random.uniform(1, 3) * (attempt + 1)) # Exponential backoff
                    
                elif error_type == "INSUFFICIENT_CREDITS":
                    if not self._rotate_api_key(f"Insufficient credits: {str(e)}"):
                        break
                    time.sleep(random.uniform(1, 3) * (attempt + 1))
                    
                elif error_type == "TIMEOUT" and attempt < max_retries - 1:
                    time.sleep(random.uniform(0.5, 1.5))
                    
                elif error_type == "INVALID_URL":
                    self.logger.error(f"Invalid URL error: {str(e)}")
                    break # Do not retry on invalid URL errors
                    
                else:
                    # Generic error handling
                    if attempt < max_retries - 1:
                        time.sleep(random.uniform(0.2, 0.5))
                    elif not self._rotate_api_key(f"Error: {str(e)}"):
                        break
                    else:
                        time.sleep(random.uniform(0.5, 2))
        
        raise Exception(f"Failed after {max_retries} attempts: {str(last_error)}")

    def scrape_url(self, url: str) -> Optional[str]:
        """
        Scrapes content from a single URL.

        Args:
            url (str): The URL of the page to scrape.

        Returns:
            Optional[str]: The raw content of the page as a string, or None if scraping fails.
        """
        try:
            if not self._validate_url(url):
                self.logger.error(f"Invalid URL provided for scraping: {url}")
                return None

            response = self._execute_with_retry(
                self.firecrawl_app.scrape,
                url=url
            )
            
            if response and 'content' in response:
                self.logger.info(f"Scraping successful for: {url}")
                return response['content']
            else:
                self.logger.warning(f"Could not retrieve content from URL: {url}")
                return None
        except Exception as e:
            self.logger.error(f"Error during scraping of URL {url}: {e}")
            return None

    def extract_structured_data(self, urls: List[str], prompt: str) -> Optional[Dict[str, Any]]:
        """
        Extracts structured data from a list of URLs using a given prompt.

        Args:
            urls (List[str]): A list of URLs from which to extract data.
            prompt (str): The prompt to guide the extraction process.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with the extracted structured data,
                                      or None if extraction fails.
        """
        try:
            valid_urls = [url for url in urls if self._validate_url(url)]
            if not valid_urls:
                self.logger.error("No valid URLs provided for structured extraction.")
                return None

            response = self._execute_with_retry(
                self.firecrawl_app.extract,
                urls=valid_urls,
                prompt=prompt,
                enable_web_search=False # Assuming URLs are already targeted
            )
            
            parsed_response = self._parse_firecrawl_response(response)
            
            if parsed_response and not parsed_response.get('error'):
                self.logger.info(f"Structured extraction successful for URLs: {', '.join(valid_urls[:3])}...")
                return parsed_response
            else:
                self.logger.warning(
                    f"Structured extraction failed for URLs: {', '.join(valid_urls[:3])}. "
                    f"Error: {parsed_response.get('error', 'Unknown')}"
                )
                return None
        except Exception as e:
            self.logger.error(f"Error during structured extraction from URLs {urls}: {e}")
            return None

    def search_and_scrape(self, query: str, num_pages: int = 1) -> List[Dict[str, Any]]:
        """
        Performs a web search and scrapes content from the results.

        Args:
            query (str): The search query.
            num_pages (int): The number of search result pages to process. Defaults to 1.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each with 'url' and 'content'
                                  from the scraped search results. Returns an empty list if failed.
        """
        try:
            # Firecrawl's .search() method directly performs search and scrapes
            response = self._execute_with_retry(
                self.firecrawl_app.search,
                query=query,
                page_limit=num_pages
            )

            results = []
            if response and isinstance(response, list):
                for item in response:
                    if 'url' in item and 'content' in item:
                        results.append({
                            'url': item['url'],
                            'content': item['content']
                        })
                self.logger.info(f"Search and scraping successful for query: '{query}'. Obtained {len(results)} results.")
            else:
                self.logger.warning(f"No search results obtained for query: '{query}'")
            return results
        except Exception as e:
            self.logger.error(f"Error during search and scraping for query '{query}': {e}")
            return []

