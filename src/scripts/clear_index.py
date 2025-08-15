"""
Elimina del índice de Vertex AI Vector Search todos los datos poblados desde la API.

Este script realiza las siguientes acciones:
1.  Se autentica contra la API para obtener un token de sesión.
2.  Consume la API de posts para obtener los mismos datos que `populate_vdb.py`.
3.  Reconstruye los IDs de los documentos tal como fueron insertados.
4.  Llama al método `remove_documents_by_ids` para eliminar los datos del índice.
"""

import os
import sys
import logging
import time
from typing import List

# Agregar el directorio raíz al path para poder importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar la lógica de la API desde el script de población y el gestor de Vector Search
from populate_vdb import get_auth_token, fetch_data, POSTS_API_URL, SEARCH_TERMS, ID_PREFIX
from modules.vector_search import VectorSearchManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_document_ids_from_api() -> List[str]:
    """
    Obtiene todos los posts de la API, reconstruye sus IDs de documento
    y devuelve una lista de todos los IDs únicos.
    """
    logger.info("Iniciando la obtención de IDs desde la API para la eliminación.")
    
    auth_token = get_auth_token()
    if not auth_token:
        logger.error("No se pudo obtener el token de autenticación. Abortando.")
        return []

    all_ids = []
    logger.info("=== Obteniendo Posts de la API para construir los IDs a eliminar ===")
    for term in SEARCH_TERMS:
        posts = fetch_data(POSTS_API_URL, term, auth_token)
        if posts:
            # Reconstruye los IDs exactamente como en populate_vdb.py
            ids = [f"{ID_PREFIX}post_{post['id']}" for post in posts]
            all_ids.extend(ids)
            logger.info(f"Se construyeron {len(ids)} IDs para el término de búsqueda '{term}'.")
    
    # Eliminar duplicados en caso de que un post aparezca en múltiples búsquedas
    unique_ids = list(set(all_ids))
    logger.info(f"Total de IDs únicos a eliminar: {len(unique_ids)}.")
    return unique_ids

def main():
    """
    Función principal para ejecutar el script de limpieza del índice.
    """
    logger.info("--- Iniciando el proceso para limpiar el índice de Vector Search ---")

    try:
        vector_manager = VectorSearchManager()
        logger.info("✅ Cliente de Vector Search inicializado.")
    except Exception as e:
        logger.error(f"No se pudo inicializar VectorSearchManager. Error: {e}", exc_info=True)
        return

    # Paso 1: Obtener los IDs de los documentos que queremos borrar.
    doc_ids_to_delete = get_all_document_ids_from_api()

    if not doc_ids_to_delete:
        logger.warning("⚠️ No se encontraron IDs para eliminar. Finalizando el proceso.")
        return

    # Paso 2: Llamar al método para eliminar los documentos.
    logger.info(f"\n=== Intentando eliminar {len(doc_ids_to_delete)} documentos del índice... ===")
    success = vector_manager.remove_documents_by_ids(doc_ids_to_delete)

    if success:
        logger.info("✅ Proceso de eliminación solicitado exitosamente.")
        logger.info("Dando 60 segundos para que la operación se procese en Google Cloud...")
        time.sleep(60) # Esperar a que la operación tenga efecto.
    else:
        logger.error("❌ Ocurrió un error durante el proceso de eliminación.")
        return # Salir si la solicitud falló.
        
    logger.info("\n=== Verificando estado final del índice... ===")
    try:
        final_stats = vector_manager.get_index_stats()
        if final_stats:
            logger.info(f"📊 Vectores totales restantes en el índice: {final_stats.get('vectors_count', 'N/A')}")
    except Exception as e:
        logger.error(f"No se pudo verificar el estado final del índice: {e}")

if __name__ == "__main__":
    main()