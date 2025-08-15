import sys
import os
import logging
import pprint
from google.cloud import aiplatform_v1

# Permite al script encontrar los módulos en el directorio 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.vector_search import VectorSearchManager

# Configuración básica para ver los logs en la consola
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_status():
    """
    Inicializa el VectorSearchManager y obtiene las estadísticas del índice.
    """
    try:
        logger.info("Inicializando VectorSearchManager para revisar el estado del índice...")
        manager = VectorSearchManager()
        
        stats = manager.get_index_stats()
        
        if stats:
            logger.info("Estadísticas del índice obtenidas exitosamente:")
            pprint.pprint(stats)
            
            if stats.get("vectors_count", 0) > 0:
                logger.info("\n✅ ¡Hay datos en el índice! El problema probablemente está en la consulta de búsqueda.")
            else:
                logger.warning("\n❌ El índice está vacío. Debes ejecutar el script de población (populate_vdb.py).")
        else:
            logger.error("No se pudieron obtener las estadísticas del índice.")

    except Exception as e:
        logger.error(f"Ocurrió un error al revisar el estado del índice: {e}", exc_info=True)

def check_index_configuration():
    # Configurar el cliente
    api_endpoint = "us-central1-aiplatform.googleapis.com"  # Ajusta según tu región
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform_v1.IndexServiceClient(client_options=client_options)
    
    # Tu índice actual
    index_path = "projects/redmag-chatbot/locations/us/indexes/5909375821016989696"
    
    try:
        # Obtener información del índice
        index_info = client.get_index(name=index_path)
        
        print("=== CONFIGURACIÓN DEL ÍNDICE ===")
        print(f"Nombre: {index_info.display_name}")
        print(index_info)
        print(f"Estado: {index_info.state}")
        
        # Extraer configuración de metadatos
        if hasattr(index_info, 'metadata') and index_info.metadata:
            metadata = dict(index_info.metadata)
            
            # Buscar configuración de dimensiones
            if 'config' in metadata:
                config = metadata['config']
                if isinstance(config, dict):
                    dimensions = config.get('dimensions', 'No especificado')
                    print(f"Dimensiones configuradas: {dimensions}")
                    
                    # Otros parámetros importantes
                    distance_measure = config.get('distanceMeasureType', 'No especificado')
                    print(f"Medida de distancia: {distance_measure}")
                    
                    algorithm = config.get('algorithmConfig', {})
                    if 'treeAhConfig' in algorithm:
                        print("Algoritmo: Tree AH")
                    elif 'bruteForceConfig' in algorithm:
                        print("Algoritmo: Brute Force")
            
            print(f"\nMetadatos completos:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        
        # Información del método de actualización
        if hasattr(index_info, 'index_update_method'):
            if index_info.index_update_method == aiplatform_v1.Index.IndexUpdateMethod.BATCH_UPDATE:
                print(f"Método de actualización: BATCH_UPDATE")
            elif index_info.index_update_method == aiplatform_v1.Index.IndexUpdateMethod.STREAM_UPDATE:
                print(f"Método de actualización: STREAM_UPDATE")
        
        # Estadísticas
        if hasattr(index_info, 'index_stats') and index_info.index_stats:
            print(f"Vectores en el índice: {index_info.index_stats.vectors_count}")
        
        return index_info
        
    except Exception as e:
        print(f"Error al obtener información del índice: {e}")
        return None

check_status()

check_index_configuration()