"""
Script para poblar la base de datos vectorial de MentorIA Chatbot.

Este script realiza las siguientes acciones:
1. Se autentica contra la API para obtener un token de sesión
2. Consume la API de posts usando el token
3. Si un post contiene un enlace externo, utiliza Firecrawl para extraer su contenido
4. Procesa y enriquece los datos de ambas fuentes
5. Inserta los documentos procesados en el índice de Vector Search
"""

import os
import sys
import requests
import logging
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import time
import json
from datetime import datetime
from google.cloud import storage

# Agregar el directorio raíz al path para poder importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapters.vector_search_adapter import VectorSearchAdapter
from src.config import config
from src.turbo_firecrawl.firecrawl_scraper import EnhancedFirecrawlClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes de configuración
API_DOMAIN = os.getenv("API_DOMAIN", "cmsv1.redmagisterial.com")
LOGIN_URL = f"https://{API_DOMAIN}/api/auth/login"
POSTS_API_URL = f"https://{API_DOMAIN}/api/v1/posts/meds"
PROMPTS_API_URL = f"https://{API_DOMAIN}/api/v1/chatgpt/prompt"
SEARCH_TERMS = ["diagnóstico", "Insumos NEM", "LTG"]
ID_PREFIX = "prod_"


def get_auth_token() -> Optional[str]:
    """Se autentica en la API para obtener un token de autorización."""
    logger.info("Intentando obtener token de autenticación...")
    try:
        payload = {"username": config.api_username, "password": config.api_password}
        response = requests.post(LOGIN_URL, json=payload, timeout=30)
        response.raise_for_status()
        token = response.json().get("token")
        if token:
            logger.info("✅ Token de autenticación obtenido exitosamente.")
            return token
        logger.error("El login fue exitoso pero no se encontró un token en la respuesta.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al intentar autenticarse en {LOGIN_URL}: {e}")
        return None


def fetch_data(url: str, search_term: str, auth_token: str) -> List[Dict[str, Any]]:
    """Obtiene todos los datos de una URL paginada de la API usando un token."""
    all_results = []
    current_url = url
    params = {'search': search_term}
    headers = {'Authorization': f'Token {auth_token}'}
    logger.info(f"Iniciando la obtención de datos para '{search_term}' desde {url}")
    
    while current_url:
        try:
            response = requests.get(current_url, params=params, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if results:
                all_results.extend(results)
                logger.info(f"Obtenidos {len(results)} resultados de {current_url}")
            params = None
            current_url = data.get("next")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al contactar la API en {current_url}: {e}")
            break
    logger.info(f"Obtención finalizada para '{search_term}'. Total: {len(all_results)} resultados.")
    return all_results


def extract_first_valid_external_url(text: str, base_domain: str) -> Optional[str]:
    """Extrae la primera URL externa válida que no sea un archivo multimedia."""
    if not text or not isinstance(text, str): 
        return None
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    potential_urls = re.findall(url_pattern, text)
    ignored_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.doc', '.zip']
    for url in potential_urls:
        try:
            parsed_url = urlparse(url)
            if parsed_url.scheme in ['http', 'https'] and not any(parsed_url.path.lower().endswith(ext) for ext in ignored_extensions):
                return url
        except Exception:
            continue
    return None


def prepare_posts_for_upsert(posts: List[Dict[str, Any]], firecrawl_client: Optional[EnhancedFirecrawlClient], search_term: str) -> List[Dict[str, Any]]:
    """Transforma posts para la inserción, enriqueciendo con web scraping."""
    documents = []
    source = f"api_{search_term.lower().replace(' ', '_')}"
    
    storage_client = storage.Client(project=config.project_id)
    content_bucket_name = config.gcs_bucket_name
    
    try:
        bucket = storage_client.bucket(content_bucket_name)
        logger.info(f"Usando bucket existente para contenido: {content_bucket_name}")
    except Exception as e:
        logger.error(f"Error al acceder al bucket: {e}")
        raise
    
    for post in posts:
        try:
            name = post.get('name', 'Sin título')
            description = post.get('description', 'Sin descripción')
            suggestion = post.get('suggestion', '')
            content_parts = [f"Título: {name}.", f"Descripción: {description}.", f"Sugerencia: {suggestion}."]
            text_to_search_link = f"{description} {suggestion}"
            link_to_scrape = extract_first_valid_external_url(text_to_search_link, API_DOMAIN)
            
            if link_to_scrape and firecrawl_client:
                logger.info(f"Enlace externo válido encontrado en post {post['id']}: {link_to_scrape}. Extrayendo...")
                scraped_content = firecrawl_client.extract_url(url=link_to_scrape)
                if scraped_content:
                    content_parts.append(f"Resumen del contenido externo: {scraped_content}")
            
            full_content = " ".join(filter(None, content_parts))
            doc_id = f"{ID_PREFIX}post_{post['id']}"
            
            content_data = {
                'id': doc_id,
                'content': full_content,
                'metadata': {
                    'source': source, 
                    'slug': post.get('slug', ''),
                    'original_id': post['id'],
                    'type': 'post',
                    'search_term': search_term
                },
                'timestamp': datetime.now().isoformat()
            }
            
            blob_path = f"documents/{doc_id}.json"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(
                json.dumps(content_data, indent=2, ensure_ascii=False),
                content_type="application/json"
            )
            
            summary_content = f"Título: {name}. Descripción: {description[:200]}..."
            
            document_data = {
                'id': doc_id,
                'content': summary_content,
                'metadata': {'source': source, 'slug': post.get('slug', '')},
                'restricts': [
                    {
                        'namespace': 'source',
                        'allow': [source]
                    }
                ]
            }
            documents.append(document_data)
        except Exception as e:
            logger.error(f"[Posts] Error procesando el post ID {post.get('id', 'N/A')}: {e}")
    return documents


def main():
    """Función principal para ejecutar el script de población de la base de datos vectorial."""
    logger.info("--- Iniciando el proceso para poblar la base de datos vectorial ---")
    
    auth_token = get_auth_token()
    if not auth_token:
        logger.error("No se pudo obtener el token de autenticación. Abortando proceso.")
        return

    try:
        vector_manager = VectorSearchAdapter()
        firecrawl_client = EnhancedFirecrawlClient(api_keys=config.firecrawl_api_keys)
        logger.info("✅ Clientes de Vector Search y Firecrawl inicializados.")
    except Exception as e:
        logger.error(f"No se pudieron inicializar los clientes. Error: {e}", exc_info=True)
        return

    index_info = vector_manager.get_index_info()
    update_method = index_info.get("update_method") if index_info else "BATCH_UPDATE"
    logger.info(f"El índice está configurado para: {update_method}. Se usará el método de inserción apropiado.")

    all_docs_to_insert = []

    logger.info("\n=== Fase 1: Procesando Posts de la API ===")
    for term in SEARCH_TERMS:
        posts = fetch_data(POSTS_API_URL, term, auth_token)
        if posts:
            documents = prepare_posts_for_upsert(posts, firecrawl_client, term)
            all_docs_to_insert.extend(documents)
    
    if not all_docs_to_insert:
        logger.warning("⚠️ No se encontraron documentos para insertar. Finalizando.")
        return

    logger.info(f"\n=== Fase 3: Insertando un total de {len(all_docs_to_insert)} documentos en Vertex AI ===")
    try:
        if update_method == "STREAM_UPDATE":
            logger.info("Usando método de inserción por Streaming...")
            results = vector_manager.insert_documents_stream(all_docs_to_insert)
        else:
            logger.info("Usando método de inserción por Lotes (Batch)...")
            results = vector_manager.insert_documents_batch(all_docs_to_insert)
        
        successful_inserts = sum(1 for status in results.values() if status)
        logger.info(f"✅ Proceso de inserción completado. Documentos exitosos: {successful_inserts} de {len(all_docs_to_insert)}")
        
        logger.info("\n=== Verificando estado final del índice... ===")
        time.sleep(5)
        final_stats = vector_manager.get_index_stats()
        if final_stats:
            logger.info(f"📊 Vectores totales en el índice: {final_stats.get('vectors_count', 'N/A')}")
        
    except Exception as e:
        logger.error(f"❌ Falló la inserción en lote en Vertex AI. Error: {e}", exc_info=True)

    logger.info("\n--- Proceso de población de VDB finalizado ---")


if __name__ == "__main__":
    main() 