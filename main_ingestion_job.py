import os
import sys
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Awaitable
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
from firecrawl_scraper import EnhancedFirecrawlClient
from firecrawl_scraper import setup_logging

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
            {"id": "db_doc_1", "title": "PostgreSQL is great", "urls":[ "https://www.postgresql.org"], "text": "PostgreSQL is a powerful, open source object-relational database system.", "created_at": datetime.now().isoformat()},
            {"id": "db_doc_2", "title": "PostgreSQL Features", "urls": ["https://www.postgresql.org/features/"], "text": "PostgreSQL has a strong reputation for reliability, data integrity, and correctness.", "created_at": datetime.now().isoformat()}
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

    async def _generate_summary(self, text: str) -> str:
        """
        Generates a concise summary of the provided text using the Gemini API.

        This method is a placeholder and requires an actual implementation using
        the Gemini API client. It's marked as async to simulate an API call.
        """
        self.logger.info("Generando resumen del texto...")
        
        # This is where the actual call to the Gemini API would be made.
        # For now, we'll return a placeholder summary.
        # You would replace this with the actual API call logic.
        #
        # A rough implementation would look something like this:
        # prompt = f"Genera un resumen conciso del siguiente texto en español:\n\n{text[:2000]}"
        # response = await call_gemini_api(prompt)
        # return response['summary']
        #
        # For demonstration purposes, we will just return a simple summary.
        return f"Resumen generado del contenido. Longitud: {len(text)} caracteres."

    def get_content(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrapes content from a provided list of URLs and generates a summary for each.
        """
        self.logger.info(f"Scraping y resumiendo contenido de {len(urls)} URLs...")
        
        content_items = []
        # Use an event loop for async operations in a non-async context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        future_to_url = {
            self.thread_pool.submit(self.firecrawl_client.extract_url, url): url for url in urls
        }
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                content = future.result()
                if content:
                    # Generate the summary asynchronously for each content item
                    summary = loop.run_until_complete(self._generate_summary(content))

                    content_items.append({
                        "id": url,
                        "title": url, # Usar la URL como título si no hay otro disponible
                        "url": url,
                        "text": content,
                        "summary": summary,  # Agregar el resumen aquí
                        "created_at": datetime.now().isoformat()
                    })
                    self.logger.info(f"Contenido y resumen procesados para la URL: {url}")
                else:
                    self.logger.warning(f"No se pudo obtener contenido para la URL: {url}")
            except Exception as e:
                self.logger.error(f"Error al procesar la URL {url}: {e}")
        
        loop.close()
        self.logger.info(f"Scraping y resúmenes completados: Se obtuvieron {len(content_items)} elementos.")
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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            summary = loop.run_until_complete(self._generate_summary(content))
            loop.close()

            self.logger.info(f"Contenido obtenido para la URL: {content_id}")
            return {
                "id": content_id,
                "title": content_id, # Usar la URL como título si no hay otro disponible
                "url": content_id,
                "text": content,
                "summary": summary,
                "created_at": datetime.now().isoformat()
            }
        else:
            self.logger.warning(f"No se pudo obtener contenido para la URL: {content_id}")
            return None

    def get_last_modified_content(self, last_sync: datetime) -> List[Dict[str, Any]]:
        # Este conector no soporta la funcionalidad de 'última modificación' para URLs arbitrarias.
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
        Prepares content items for vector search insertion, including the summary in metadata.
        """
        documents = []
        for item in content_items:
            doc_id = item.get("id") or item.get("url")
            if not doc_id:
                logger.warning(f"Skipping item due to missing ID or URL: {item}")
                continue

            # Este es el paso crítico: creamos el texto que será incrustado.
            # Incluimos el título y el contenido, pero también el resumen para una mejor
            # representación semántica.
            text_to_embed = f"Title: {item.get('title', '')}\n\nSummary: {item.get('summary', '')}\n\nContent: {item.get('text', '')}"
            
            documents.append({
                "id": str(doc_id),
                "content": text_to_embed,
                "metadata": {
                    "source": "web_scraper",
                    "original_url": item.get('url', ''),
                    "title": item.get('title', ''),
                    # Almacenamos el resumen en los metadatos como una lista de JSON, como se solicitó.
                    "summaries": [{"url": item.get('url'), "summary": item.get('summary')}],
                    "last_sync": datetime.now().isoformat()
                }
            })
        return documents

    def run_migration(self, urls_to_ingest: List[str], batch_size: int = 100) -> Dict[str, Any]:
        """
        Executes the content migration process with a list of URLs to ingest.
        """
        try:
            logger.info("Iniciando el trabajo de migración de contenido...")

            # 1. Conectar a la fuente de contenido
            if not self.connector.connect():
                logger.error("No se pudo conectar a la fuente de contenido.")
                return {"success": False, "error": "Connection failed"}

            # 2. Obtener el contenido a partir de las URLs proporcionadas
            content_items = self.connector.get_content(urls_to_ingest)
            if not content_items:
                logger.warning("No hay elementos de contenido para procesar. Saliendo.")
                return {"success": True, "updates": 0, "total_updated_items": 0}

            # 3. Preparar los documentos para la inserción en el vector search
            documents = self._prepare_documents(content_items)

            # 4. Insertar los documentos en la base de datos vectorial
            results = self.vector_manager.upsert_documents_batch(documents, batch_size)
            
            # 5. Calcular estadísticas
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
            
            logger.info(f"Migración completada: {successful} éxitos, {failed} fallos")
            return migration_stats
            
        except Exception as e:
            logger.error(f"La migración falló: {e}")
            return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Para ejecutar este script, asegúrate de que las variables de entorno (en un archivo .env)
    # estén correctamente configuradas.
    # FIRECRAWL_API_KEYS="your-firecrawl-key-1,your-firecrawl-key-2"
    # El resto de las configuraciones de GCP están en config.py

    # Y que las bibliotecas de `requirements.txt` estén instaladas:
    # pip install -r requirements.txt
    # pip install firecrawl-py
    # pip install httpx # Se recomienda para llamadas async

    # Simula la lista de URLs que obtendrías de tu base de datos.
    urls_from_db = [
        "https://www.who.int/es/news-room/fact-sheets/detail/mental-health-strengthening-our-response",
        "https://www.mentalhealth.org.uk/about/what-is-mental-health",
        "https://www.paho.org/es/temas/salud-mental"
    ]

    try:
        # Obtener las claves de API de Firecrawl de las variables de entorno
        firecrawl_keys = os.getenv("FIRECRAWL_API_KEYS")
        if not firecrawl_keys:
            raise ValueError("La variable de entorno FIRECRAWL_API_KEYS no está configurada.")
        
        api_keys_list = [key.strip() for key in firecrawl_keys.split(',')]
        
        # Inicializar componentes
        firecrawl_connector = FirecrawlScraperConnector(api_keys=api_keys_list)
        vector_search_manager = VectorSearchManager()
        
        # Inicializar y ejecutar el trabajo de migración
        job = MigrationJob(connector=firecrawl_connector, vector_manager=vector_search_manager)
        migration_results = job.run_migration(urls_to_ingest=urls_from_db, batch_size=config.batch_size)
        
        print("\n=== Resultado de la migración ===")
        print(json.dumps(migration_results, indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"Error fatal al ejecutar el trabajo de migración: {e}")
