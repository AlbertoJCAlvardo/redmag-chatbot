"""
Configuration module for the Chatbot application.
"""

import os
import json
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for all project settings."""
    
    def __init__(self):
        """Initialize configuration with environment variables."""
        # Google Cloud Configuration
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east1")
        self.service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        self.gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")

        # BigQuery Configuration
        self.bigquery_users_table = os.getenv("BIGQUERY_USERS_TABLE", "redmag-chatbot.redmag_chatbot_dataset_prod.users")
        self.bigquery_messages_table = os.getenv("BIGQUERY_MESSAGES_TABLE", "redmag-chatbot.redmag_chatbot_dataset_prod.messages")
        self.bigquery_context_table = os.getenv("BIGQUERY_CONTEXT_TABLE", "redmag-chatbot.redmag_chatbot_dataset_prod.conversation_context")
        
        # Vertex AI Vector Search Configuration
        self.index_id = os.getenv("VECTOR_INDEX_ID")
        self.endpoint_id = os.getenv("VECTOR_ENDPOINT_ID")
        self.deployed_index_id = os.getenv("DEPLOYED_INDEX_ID")
        
        # Embedding Model Configuration
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
        
        # Firecrawl Scraper Configuration
        self.firecrawl_api_keys = [os.getenv("FIRECRAWL_API_KEYS")] 
        
        # Gemini AI Configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # API Authentication (for external CMS APIs)
        self.api_username = os.getenv("API_USERNAME")
        self.api_password = os.getenv("API_PASSWORD")
        
        self.batch_size = 100
        
        # Logging Configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.static_files_path = Path(os.getenv("STATIC_FILES_PATH"))
        self.knowledge_base_nem_path = self.static_files_path /     Path(os.getenv("KNOWLEDGE_BASE_NEM_PATH"))
        with open(self.knowledge_base_nem_path, "r") as f:
            self.knowledge_base_nem = json.load(f)
            f.close()
        if self.knowledge_base_nem is None:
            raise ValueError("Knowledge base NEM not found")
        self.sep_knowledge_base_path = self.static_files_path / Path(os.getenv("SEP_KNOWLEDGE_BASE_PATH"))
        with open(self.sep_knowledge_base_path, "r") as f:
            self.sep_knowledge_base = json.load(f)
            f.close()
        if self.sep_knowledge_base is None:
            raise ValueError("SEP knowledge base not found")

    def validate(self) -> bool:
        """Validate that required configuration values are present."""
        required_fields = ["project_id", "index_id", "bigquery_users_table"]
        missing_fields = [field for field in required_fields if not getattr(self, field)]
        if missing_fields:
            raise ValueError(f"Missing required configuration: {missing_fields}")
        return True
    
    def get_index_path(self) -> str:
        """Get the full path for the vector index."""
        return f"projects/{self.project_id}/locations/us/indexes/{self.index_id}"
    
    def get_endpoint_path(self) -> Optional[str]:
        """Get the full path for the vector endpoint if available."""
        if self.endpoint_id:
            return f"projects/{self.project_id}/locations/{self.location}/indexEndpoints/{self.endpoint_id}"
        return None
    
    def setup_authentication(self) -> None:
        """Setup Google Cloud authentication using service account."""
        if not self.service_account_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not configured")
        
        service_account_file = Path(self.service_account_path)
        if not service_account_file.exists():
            raise FileNotFoundError(f"Service account file not found: {self.service_account_path}")
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(service_account_file.absolute())

# Global configuration instance
config = Config()
