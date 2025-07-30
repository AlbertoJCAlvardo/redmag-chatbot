import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import logging

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

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

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
        logger.info("PostgreSQLConnector initialized (placeholder for real DB connection).")
        self._simulated_data = self._load_simulated_data() # For demonstration

    def connect(self) -> bool:
        """
        Connects to the PostgreSQL database.
        In a real scenario, this would test the database connection.
        """
        try:
            # Example: self.Session().connection()
            logger.info("Successfully simulated connection to PostgreSQL database.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL database: {e}")
            return False

    def _load_simulated_data(self) -> List[Dict[str, Any]]:
        """
        Loads simulated data representing content fetched from the PostgreSQL database.
        This data structure mimics the output of the SQL query provided previously.
        """
        simulated_db_content = []

        # Example data mimicking the SQL query output for various content types
        simulated_db_content.append({
            "id": "med_diagnostico_001",
            "created_date": datetime(2023, 10, 20).isoformat(),
            "last_modified": datetime(2023, 10, 26).isoformat(),
            "content_name": "Herramientas para Examen Diagnóstico",
            "content_description": "Estos MED contienen diferentes herramientas para que los docentes evalúen a sus alumnos, su escuela y su comunidad para lograr la lectura de la realidad necesaria para la NEM.",
            "content_suggestion": "Ideal para el inicio del ciclo escolar para conocer a tu grupo.",
            "image_url": "https://placehold.co/150x100/AEC6CF/000000?text=MED+Diagnóstico",
            "external_url": "https://redmagisterial.com/meds?busqueda=Diagn%C3%B3stico",
            "user_id": 101,
            "downloads": 500,
            "curricular_content_name": "Diagnóstico Inicial",
            "formative_field_name": "Lenguaje y Comunicación",
            "grade_name": "Primaria",
            "level_name": "Básico",
            "phase_name": "Fase 3",
            "subject_name": "Español",
            "learning_progression_name": "Evaluación formativa",
            "type": "med" # Explicitly add type for processing
        })

        simulated_db_content.append({
            "id": "tool_planea_magia_001",
            "created_date": datetime(2024, 1, 1).isoformat(),
            "last_modified": datetime(2024, 1, 5).isoformat(),
            "content_name": "Generador de Planeaciones Red Magisterial",
            "content_description": "El reconocido generador de planeaciones de Red Magisterial que es un aliado para docentes de todos los niveles de Educación Básica.",
            "content_suggestion": "Crea tus planeaciones didácticas de forma rápida y eficiente.",
            "image_url": "https://placehold.co/150x100/D4EDDA/000000?text=Planea+MagIA",
            "external_url": "https://redmagisterial.com/red-magia/planea",
            "user_id": 103,
            "downloads": 1200,
            "curricular_content_name": "Planeación Didáctica",
            "formative_field_name": "Múltiples",
            "grade_name": "Todos",
            "level_name": "Todos",
            "phase_name": "Todas",
            "subject_name": "Todas",
            "learning_progression_name": "Herramientas de apoyo",
            "type": "tool"
        })

        simulated_db_content.append({
            "id": "ai_assistant_nem_001",
            "created_date": datetime(2024, 1, 25).isoformat(),
            "last_modified": datetime(2024, 2, 1).isoformat(),
            "content_name": "Asistente Todo sobre la NEM",
            "content_description": "Un asistente de IA que ofrece respuestas para los docentes en torno a todos los conceptos fundamentales de la NEM y cómo se implementan en el aula.",
            "content_suggestion": "Resuelve tus dudas sobre la Nueva Escuela Mexicana.",
            "image_url": "https://placehold.co/150x100/C3E6CB/000000?text=Asistente+NEM",
            "external_url": "https://redmagisterial.com/asistentes/todo-sobre-nem",
            "user_id": 104,
            "downloads": 800,
            "curricular_content_name": "Nueva Escuela Mexicana",
            "formative_field_name": "Múltiples",
            "grade_name": "Todos",
            "level_name": "Todos",
            "phase_name": "Todas",
            "subject_name": "Todas",
            "learning_progression_name": "Asistencia IA",
            "type": "ai_assistant"
        })

        simulated_db_content.append({
            "id": "webinar_w111",
            "created_date": datetime(2024, 2, 28).isoformat(),
            "last_modified": datetime(2024, 3, 1).isoformat(),
            "content_name": "W111 Del diagnóstico a la planeación multidisciplinaria en preescolar",
            "content_description": "La aplicación de los principios de la Nueva Escuela Mexicana en las aulas es un reto para los docentes",
            "content_suggestion": "Aprende a planear de forma multidisciplinaria en preescolar.",
            "image_url": "https://placehold.co/150x100/B0E0E6/000000?text=Webinar+1",
            "external_url": "https://redmagisterial.com/webinars/w111", # Fictitious URL
            "user_id": 105,
            "downloads": 300,
            "curricular_content_name": "Diagnóstico y Planeación",
            "formative_field_name": "Múltiples",
            "grade_name": "Preescolar",
            "level_name": "Básico",
            "phase_name": "Fase 1",
            "subject_name": "Múltiples",
            "learning_progression_name": "Estrategias didácticas",
            "type": "webinar"
        })

        logger.info("Simulated database content loaded.")
        return simulated_db_content

    def get_content(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves content from the CMS (simulating a DB query).
        In a real scenario, this would execute the actual SQL query via an ORM.
        """
        logger.info(f"Executing SQL query to retrieve content (limit: {limit})...")
        # Example of how you would execute the SQL query using an ORM:
        # with self.Session() as session:
        #     # Construct your ORM query based on the SQL provided previously
        #     # e.g., query = session.query(YourContentModel).filter(...).order_by(...)
        #     # if limit: query = query.limit(limit)
        #     # results = query.all()
        #     # content_items = [item.to_dict() for item in results] # Convert ORM objects to dicts
        # return content_items

        if limit:
            return self._simulated_data[:limit]
        return self._simulated_data

    def get_content_by_id(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a content item by ID (simulating a DB query).
        """
        logger.info(f"Searching for content by ID: {content_id} (simulating DB)...")
        for item in self._simulated_data:
            if item["id"] == content_id:
                return item
        return None

    def get_updated_content(self, since: datetime) -> List[Dict[str, Any]]:
        """
        Retrieves content updated since a specific date (simulating a DB query).
        """
        logger.info(f"Searching for content updated since: {since} (simulating DB)...")
        updated_items = []
        for item in self._simulated_data:
            item_modified_date = datetime.fromisoformat(item["last_modified"])
            if item_modified_date > since:
                updated_items.append(item)
        return updated_items

# --- Global Firecrawl Client Instance ---
# This will be initialized once in run_ingestion_job
firecrawl_client: Optional[EnhancedFirecrawlClient] = None

def fetch_content_from_url_with_firecrawl(url: str) -> Optional[str]:
    """
    Extracts text content from a URL using the EnhancedFirecrawlClient.

    Args:
        url (str): The URL to scrape.

    Returns:
        Optional[str]: The scraped text content, or None if an error occurs.
    """
    global firecrawl_client
    if not firecrawl_client:
        logger.error("EnhancedFirecrawlClient is not initialized.")
        return None
    
    logger.info(f"Extracting content from URL with Firecrawl: {url}")
    return firecrawl_client.scrape_url(url)

# --- Content Processing Function ---
def red_magisterial_content_processor(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Processes a CMS content item to prepare it for the vector database.
    Combines relevant text fields and structures metadata.
    Attempts to fetch external URL content using Firecrawl.

    Args:
        item (Dict[str, Any]): The content item dictionary fetched from the CMS.

    Returns:
        Optional[Dict[str, Any]]: A processed document dictionary ready for vector
                                  database insertion, or None if the item is invalid.
    """
    if not item:
        return None

    # Combine text fields for embedding
    title = item.get('content_name', '')
    description = item.get('content_description', '')
    suggestion = item.get('content_suggestion', '')
    
    combined_content = f"Title: {title}\nDescription: {description}\nUsage Suggestion: {suggestion}".strip()

    # If there's an external URL, try to extract additional content with Firecrawl
    external_url = item.get('external_url')
    if external_url:
        try:
            content_from_url = fetch_content_from_url_with_firecrawl(external_url)
            if content_from_url:
                combined_content += f"\n\nExternal URL Content: {content_from_url}"
        except Exception as e:
            logger.warning(f"Could not extract content from URL {external_url} with Firecrawl: {e}")

    if not combined_content:
        logger.warning(f"Skipping item {item.get('id', 'unknown')} - no textual content to embed.")
        return None

    # Structure metadata
    metadata = {
        'title': title,
        'description': description,
        'suggestion': suggestion,
        'url': external_url,
        'image_url': item.get('image_url', ''),
        'type': item.get('type', 'general_content'), # Use 'type' from simulated data or default
        'categories': [], # Will be populated with names from related tables
        'last_modified': item.get('last_modified', datetime.utcnow().isoformat()),
        'source': 'red_magisterial_cms_sync'
    }
    
    # Add names from related tables to categories metadata
    if item.get('curricular_content_name'):
        metadata['categories'].append(item['curricular_content_name'])
    if item.get('formative_field_name'):
        metadata['categories'].append(item['formative_field_name'])
    if item.get('grade_name'):
        metadata['categories'].append(item['grade_name'])
    if item.get('level_name'):
        metadata['categories'].append(item['level_name'])
    if item.get('phase_name'):
        metadata['categories'].append(item['phase_name'])
    if item.get('subject_name'):
        metadata['categories'].append(item['subject_name'])
    if item.get('learning_progression_name'):
        metadata['categories'].append(item['learning_progression_name'])

    # Remove duplicates in categories
    metadata['categories'] = list(set(metadata['categories']))

    return {
        'id': item['id'],
        'content': combined_content, # This is the text that will be embedded
        'metadata': metadata
    }

# --- Main Ingestion Job Function ---
def run_ingestion_job():
    """
    Executes the content ingestion job from Red Magisterial CMS
    to Vertex AI Vector Search.
    """
    global firecrawl_client # Access the global variable

    logger.info("Starting Red Magisterial content ingestion job...")

    try:
        # 1. Validate and set up the environment
        config.validate()
        config.setup_authentication()
        logger.info("Vertex AI configuration validated and authentication established.")

        # 2. Initialize EnhancedFirecrawlClient
        firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY") # Ensure this variable is in .env
        if not firecrawl_api_key:
            logger.error("FIRECRAWL_API_KEY not found in environment variables.")
            raise ValueError("FIRECRAWL_API_KEY is required for URL scraping.")
        
        # You can pass a list of keys if you have multiple for rotation
        # The log_language parameter can be set to "es" for Spanish logs in the scraper
        firecrawl_client = EnhancedFirecrawlClient(api_keys=[firecrawl_api_key], log_language="en")
        logger.info("EnhancedFirecrawlClient initialized.")

        # 3. Initialize VectorSearchManager and CMSIntegration
        vector_manager = VectorSearchManager()
        cms_integration = CMSIntegration(vector_manager)
        logger.info("VectorSearchManager and CMSIntegration initialized.")

        # 4. Initialize the Red Magisterial CMS connector (PostgreSQLConnector placeholder)
        red_magisterial_connector = PostgreSQLConnector()
        
        # 5. Perform the initial content migration
        logger.info("Starting initial content migration...")
        migration_results = cms_integration.migrate_content(
            cms_connector=red_magisterial_connector,
            content_processor=red_magisterial_content_processor,
            batch_size=config.batch_size # Use batch size from config
        )

        if migration_results["success"]:
            logger.info(
                f"Initial migration completed successfully. "
                f"Documents inserted: {migration_results['successful_insertions']}"
            )
            logger.info(f"Failed documents: {migration_results['failed_insertions']}")
        else:
            logger.error(f"Initial migration failed: {migration_results['error']}")
            return

        # 6. Simulate an update synchronization (for periodic execution)
        last_sync_time = datetime.utcnow()
        logger.info(f"Simulating next synchronization from: {last_sync_time}")

        # In a real production environment, you would call cms_integration.sync_updates here
        # sync_results = cms_integration.sync_updates(
        #     cms_connector=red_magisterial_connector,
        #     last_sync=last_sync_time,
        #     content_processor=red_magisterial_content_processor
        # )
        # if sync_results["success"]:
        #     logger.info(f"Update synchronization completed. Documents updated: {sync_results['updates']}")
        # else:
        #     logger.error(f"Update synchronization failed: {sync_results['error']}")

        logger.info("Content ingestion job finished.")

    except Exception as e:
        logger.critical(f"Critical error in ingestion job: {e}", exc_info=True)

if __name__ == "__main__":
    # To run this script, ensure your environment variables (in a .env file)
    # are correctly configured:
    # GOOGLE_CLOUD_PROJECT_ID=your-project-id
    # GOOGLE_CLOUD_LOCATION=us-central1
    # GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
    # VECTOR_INDEX_ID=your-vector-index-id
    # EMBEDDING_MODEL=textembedding-gecko@003
    # FIRECRAWL_API_KEY=your-firecrawl-api-key
    # LOG_LEVEL=INFO (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)

    # And that the libraries from `requirements.txt` are installed:
    # pip install -r requirements.txt
    # pip install firecrawl-py # Ensure Firecrawl is installed

    run_ingestion_job()
