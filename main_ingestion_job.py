import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the project root and the 'turbo-firecrawl' directory to the sys.path
# This allows importing modules from 'modules' and 'turbo-firecrawl'
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'modules'))
sys.path.append(os.path.join(project_root, 'turbo-firecrawl')) # Add turbo-firecrawl to path

# Import necessary modules
from config import config
from vector_search import VectorSearchManager
from cms_integration import CMSIntegration, CMSConnector
from firecrawl_scraper import EnhancedFirecrawlClient # Updated import path
from firecrawl_scraper import setup_logging # Import the logger setup function

# Configure logging
logger = setup_logging(lang="es")
logging.basicConfig(level=getattr(logging, config.log_level))


# --- Placeholder for a real PostgreSQL DB Connector ---
# In a real scenario, this class would connect to your PostgreSQL database
# using an ORM (e.g., SQLAlchemy, Django ORM) and execute the SQL query.
class PostgreSQLConnector(CMSConnector):
    """
    A placeholder for the PostgreSQL CMS connector.
    In a real application, this class would implement the actual database
    connection and query logic using an ORM (e.g., SQLAlchemy, Django ORM).
    """
    def __init__(self):
        # Initialize your database connection/ORM session here
        # Example: self.engine = create_engine(config.database_url)
        # self.Session = sessionmaker(bind=self.engine)
        self.logger = logging.getLogger(__name__)
        self.logger.info("PostgreSQLConnector initialized.")

    def connect(self) -> bool:
        self.logger.info("Connecting to PostgreSQL...")
        # Simulate connection
        return True

    def get_content(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        self.logger.info(f"Retrieving content from PostgreSQL (limit: {limit})...")
        # In a real app, this would execute a SQL query and fetch results.
        # Example dummy data
        return [
            {"id": "db_doc_1", "title": "PostgreSQL is great", "url": "https://www.postgresql.org", "text": "PostgreSQL is a powerful, open source object-relational database system.", "created_at": datetime.now().isoformat()},
            {"id": "db_doc_2", "title": "PostgreSQL Features", "url": "https://www.postgresql.org/features/", "text": "PostgreSQL has a strong reputation for reliability, data integrity, and correctness.", "created_at": datetime.now().isoformat()}
        ]

    def get_content_by_id(self, content_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        self.logger.info(f"Retrieving content by ID from PostgreSQL: {content_id}")
        # In a real app, this would query the database for a specific item.
        return None
        
    def get_last_modified_content(self, last_sync: datetime) -> List[Dict[str, Any]]:
        self.logger.info(f"Retrieving content modified since {last_sync} from PostgreSQL...")
        # In a real app, this would execute a query to find recent updates.
        return []


# --- Firecrawl-based Scraper Connector ---
class FirecrawlScraperConnector(CMSConnector):
    """
    A connector that uses the EnhancedFirecrawlClient to scrape web content.

    This connector simulates fetching content from a 'CMS' by scraping a list of URLs
    or performing a web search. It uses the improved `extract_url` and `search_and_extract_links`
    methods from the `EnhancedFirecrawlClient`.
    """
    def __init__(self, api_keys: List[str]):
        """
        Initializes the connector with a list of Firecrawl API keys.
        """
        self.logger = logging.getLogger(__name__)
        self.firecrawl_client = EnhancedFirecrawlClient(api_keys=api_keys, log_language="es")
        self.logger.info("FirecrawlScraperConnector initialized.")
        self.thread_pool = ThreadPoolExecutor(max_workers=5) # Use a thread pool for concurrent scraping

    def connect(self) -> bool:
        # No explicit connection needed, as the client handles authentication per request.
        self.logger.info("Firecrawl client is ready.")
        return True

    def get_content(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Performs a web search and scrapes content from the top results.

        This method is for demonstration purposes. In a real-world scenario, you might
        provide a list of known URLs to scrape instead of a search query.
        """
        self.logger.info(f"Retrieving content using Firecrawl (limit: {limit})...")
        
        # This part simulates a content source by searching for a topic.
        search_query = "Mejores prácticas de salud mental"
        search_results = self.firecrawl_client.search_and_extract_links(query=search_query, num_results=limit or 10)
        
        if not search_results:
            self.logger.error("No se encontraron resultados de búsqueda para poblar la base de datos.")
            return []

        content_items = []
        future_to_url = {
            self.thread_pool.submit(self.firecrawl_client.extract_url, item['url']): item for item in search_results
        }
        
        for future in as_completed(future_to_url):
            url_item = future_to_url[future]
            url = url_item['url']
            try:
                content = future.result()
                if content:
                    content_items.append({
                        "id": url,
                        "title": url_item.get('title', url),
                        "url": url,
                        "text": content,
                        "created_at": datetime.now().isoformat()
                    })
            except Exception as e:
                self.logger.error(f"Error al procesar la URL {url}: {e}")
        
        self.logger.info(f"Scraping completado: Se obtuvieron {len(content_items)} elementos de contenido.")
        return content_items

    def get_content_by_id(self, content_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """
        Scrapes a single URL identified by the content_id.
        """
        self.logger.info(f"Retrieving content by ID using Firecrawl: {content_id}")
        if not isinstance(content_id, str):
            self.logger.error("ID de contenido inválido. Debe ser una URL.")
            return None
        
        content = self.firecrawl_client.extract_url(content_id)
        if content:
            self.logger.info(f"Contenido obtenido para la URL: {content_id}")
            return {
                "id": content_id,
                "title": content_id, # Can't get title from URL alone, so use the URL
                "url": content_id,
                "text": content,
                "created_at": datetime.now().isoformat()
            }
        else:
            self.logger.warning(f"No se pudo obtener contenido para la URL: {content_id}")
            return None

    def get_last_modified_content(self, last_sync: datetime) -> List[Dict[str, Any]]:
        # This connector does not support tracking last modified dates easily.
        self.logger.warning("Este conector no soporta la funcionalidad de 'última modificación'.")
        return []


class MigrationJob:
    """
    Handles the end-to-end process of migrating content to the vector database.
    """
    def __init__(self, connector: CMSConnector, vector_manager: VectorSearchManager):
        """
        Initializes the migration job.

        Args:
            connector (CMSConnector): An instance of a CMS connector.
            vector_manager (VectorSearchManager): An instance of the vector search manager.
        """
        self.connector = connector
        self.vector_manager = vector_manager
        
    def _prepare_documents(self, content_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepares content items for vector search insertion.
        """
        documents = []
        for item in content_items:
            doc_id = item.get("id") or item.get("url")
            if not doc_id:
                logger.warning(f"Skipping item due to missing ID or URL: {item}")
                continue

            # This is a critical step: create the text to be embedded.
            # Combine the title, description, and text to create a rich document.
            text_to_embed = f"Title: {item.get('title', '')}\n\nDescription: {item.get('description', '')}\n\nContent: {item.get('text', '')}"
            
            documents.append({
                "id": str(doc_id),
                "content": text_to_embed,
                "metadata": {
                    "source": "web_scraper",
                    "original_url": item.get('url', ''),
                    "title": item.get('title', ''),
                    "last_sync": datetime.now().isoformat()
                }
            })
        return documents

    def run_migration(self, batch_size: int = 100) -> Dict[str, Any]:
        """
        Executes the content migration process.
        """
        try:
            logger.info("Starting content migration job...")

            # 1. Connect to CMS (or content source)
            if not self.connector.connect():
                logger.error("Failed to connect to the content source.")
                return {"success": False, "error": "Connection failed"}

            # 2. Retrieve content
            content_items = self.connector.get_content()
            if not content_items:
                logger.warning("No content items to process. Exiting.")
                return {"success": True, "updates": 0, "total_updated_items": 0}

            # 3. Prepare documents for vector search
            documents = self._prepare_documents(content_items)

            # 4. Upsert documents into the vector database
            results = self.vector_manager.upsert_documents_batch(documents, batch_size)
            
            # 5. Calculate statistics
            successful = len([r for r in results.values() if r])
            failed = len(results) - successful
            
            migration_stats = {
                "success": True,
                "total_content_items": len(content_items),
                "total_documents": len(documents),
                "successful_insertions": successful,
                "failed_insertions": failed,
                "results": results
            }
            
            logger.info(f"Migration completed: {successful} successful, {failed} failed")
            return migration_stats
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # To run this script, ensure your environment variables (in a .env file)
    # are correctly configured:
    # GOOGLE_CLOUD_PROJECT_ID=your-project-id
    # GOOGLE_CLOUD_LOCATION=us-central1
    # GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
    # VECTOR_INDEX_ID=your-vector-index-id
    # EMBEDDING_MODEL=textembedding-gecko@003
    # FIRECRAWL_API_KEYS="your-firecrawl-key-1,your-firecrawl-key-2"
    # LOG_LEVEL=INFO (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)

    # And that the libraries from `requirements.txt` are installed:
    # pip install -r requirements.txt
    # pip install firecrawl-py

    # Use the FirecrawlScraperConnector to ingest data from the web.
    try:
        # Get the API keys from environment variables
        firecrawl_keys = os.getenv("FIRECRAWL_API_KEYS")
        if not firecrawl_keys:
            raise ValueError("FIRECRAWL_API_KEYS environment variable is not set.")
        
        api_keys_list = [key.strip() for key in firecrawl_keys.split(',')]
        
        # Initialize components
        firecrawl_connector = FirecrawlScraperConnector(api_keys=api_keys_list)
        vector_search_manager = VectorSearchManager()
        
        # Initialize and run the migration job
        job = MigrationJob(connector=firecrawl_connector, vector_manager=vector_search_manager)
        migration_results = job.run_migration(batch_size=config.batch_size)
        
        print("\n=== Resultado de la migración ===")
        print(json.dumps(migration_results, indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"Error fatal al ejecutar el trabajo de migración: {e}")
