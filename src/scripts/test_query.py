import sys
import os
import logging
import pprint

# Permite al script encontrar los módulos en el directorio 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.vector_search import VectorSearchManager
from modules.config import config

# Configuración básica para ver los logs en la consola
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_query(query: str, num_results: int = 5):
    """
    Ejecuta una consulta específica y muestra los resultados.
    
    Args:
        query: La consulta a ejecutar
        num_results: Número de resultados a mostrar
    """
    try:
        logger.info("Inicializando VectorSearchManager para la consulta...")
        manager = VectorSearchManager()
        
        # Verificación de configuración
        if not manager.endpoint_path or not config.deployed_index_id:
            logger.error("El endpoint del índice (index_endpoint_id) o el ID del índice desplegado (deployed_index_id) no están configurados.")
            return
        
        logger.info(f"Ejecutando búsqueda para: '{query}'")
        
        # Búsqueda con contenido incluido
        results = manager.search_similar(query=query, num_neighbors=num_results, include_content=True)
        
        if results:
            logger.info(f"Se encontraron {len(results)} documentos similares:")
            logger.info("=" * 80)
            
            for i, result in enumerate(results, 1):
                print(f"\n📄 DOCUMENTO #{i}")
                print("-" * 40)
                print(f"ID: {result['id']}")
                print(f"Distancia: {result['distance']}")
                
                # Mostrar información sobre el documento
                print(f"Información disponible del documento:")
                print(f"  - ID: {result['id']}")
                print(f"  - Distancia de similitud: {result['distance']}")
                
                if 'metadata' in result and result['metadata']:
                    print(f"  - Metadatos disponibles:")
                    pprint.pprint(result['metadata'], width=80, depth=2)
                else:
                    print(f"  - Metadatos: No disponibles")
                
                # Intentar obtener el contenido completo desde GCS
                print(f"  - Recuperando contenido completo desde GCS...")
                doc_details = manager.get_document_by_id(result['id'])
                if doc_details and 'content' in doc_details:
                    print(f"  - Contenido completo del documento:")
                    print(f"    {doc_details['content'][:500]}...")  # Mostrar primeros 500 caracteres
                    if 'metadata' in doc_details:
                        print(f"  - Metadatos completos:")
                        pprint.pprint(doc_details['metadata'], width=80, depth=2)
                else:
                    print(f"  - Contenido: No disponible en GCS")
                
                print("-" * 40)
        else:
            logger.info("No se encontraron documentos similares para esta consulta.")
            
    except Exception as e:
        logger.error(f"Error durante la consulta: {e}", exc_info=True)


def run_test_query():
    """
    Inicializa el VectorSearchManager y ejecuta consultas de prueba
    contra el índice de Vector Search desplegado.
    """
    try:
        logger.info("Inicializando VectorSearchManager para la consulta...")
        manager = VectorSearchManager()
        
        # Lista de consultas de prueba para evaluar diferentes aspectos
        test_queries = [
            "explícame qué son los libros de texto gratuitos y cómo se usan",
            "diagnóstico educativo",
            "insumos NEM",
            "LTG evaluación",
            "planeación didáctica"
        ]
        
        num_results = 5
        
        # Verificación de que la configuración necesaria para la búsqueda está presente
        if not manager.endpoint_path or not config.deployed_index_id:
            logger.error("El endpoint del índice (index_endpoint_id) o el ID del índice desplegado (deployed_index_id) no están configurados. Por favor, revísalos en tu archivo de configuración.")
            return
        
        for i, test_query in enumerate(test_queries, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"CONSULTA #{i}: '{test_query}'")
            logger.info(f"{'='*80}")
            
            logger.info(f"Ejecutando búsqueda para la consulta: '{test_query}'")
            
            # Llama al método de búsqueda con contenido incluido
            results = manager.search_similar(query=test_query, num_neighbors=num_results, include_content=True)
            
            if results:
                logger.info(f"Se encontraron {len(results)} documentos similares:")
                logger.info("=" * 80)
                
                for j, result in enumerate(results, 1):
                    print(f"\n📄 DOCUMENTO #{j}")
                    print("-" * 40)
                    print(f"ID: {result['id']}")
                    print(f"Distancia: {result['distance']}")
                    
                                    # Mostrar información sobre el documento
                print(f"Información disponible del documento:")
                print(f"  - ID: {result['id']}")
                print(f"  - Distancia de similitud: {result['distance']}")
                
                if 'metadata' in result and result['metadata']:
                    print(f"  - Metadatos disponibles:")
                    pprint.pprint(result['metadata'], width=80, depth=2)
                else:
                    print(f"  - Metadatos: No disponibles")
                
                # Intentar obtener el contenido completo desde GCS
                print(f"  - Recuperando contenido completo desde GCS...")
                doc_details = manager.get_document_by_id(result['id'])
                if doc_details and 'content' in doc_details:
                    print(f"  - Contenido completo del documento:")
                    print(f"    {doc_details['content'][:500]}...")  # Mostrar primeros 500 caracteres
                    if 'metadata' in doc_details:
                        print(f"  - Metadatos completos:")
                        pprint.pprint(doc_details['metadata'], width=80, depth=2)
                else:
                    print(f"  - Contenido: No disponible en GCS")
                    
                    print("-" * 40)
            else:
                logger.info("No se encontraron documentos similares para esta consulta.")
            
            # Pausa entre consultas para mejor legibilidad
            if i < len(test_queries):
                print("\n" + "="*80)
                print("Presiona Enter para continuar con la siguiente consulta...")
                input()

    except Exception as e:
        logger.error(f"Ocurrió un error durante la ejecución de la consulta de prueba: {e}", exc_info=True)

if __name__ == "__main__":
    print("🔍 TEST QUERY - Vector Search")
    print("=" * 50)
    print("1. Ejecutar consultas de prueba predefinidas")
    print("2. Ejecutar consulta personalizada")
    print("3. Listar documentos almacenados en GCS")
    print("4. Mostrar estructura del bucket")
    print("5. Salir")
    
    choice = input("\nSelecciona una opción (1-5): ").strip()
    
    if choice == "1":
        run_test_query()
    elif choice == "2":
        custom_query = input("Ingresa tu consulta: ").strip()
        if custom_query:
            num_results = input("Número de resultados (default: 5): ").strip()
            num_results = int(num_results) if num_results.isdigit() else 5
            test_single_query(custom_query, num_results)
        else:
            print("Consulta vacía. Saliendo...")
    elif choice == "3":
        try:
            manager = VectorSearchManager()
            documents = manager.list_stored_documents(limit=20)
            if documents:
                print(f"\n📚 Documentos almacenados en GCS ({len(documents)} encontrados):")
                print("=" * 80)
                for i, doc in enumerate(documents, 1):
                    print(f"\n📄 DOCUMENTO #{i}")
                    print("-" * 40)
                    print(f"ID: {doc['id']}")
                    print(f"Tipo: {doc['type']}")
                    print(f"Fuente: {doc['source']}")
                    print(f"Timestamp: {doc['timestamp']}")
                    print(f"Vista previa: {doc['content_preview']}")
                    print("-" * 40)
            else:
                print("No se encontraron documentos almacenados en GCS.")
        except Exception as e:
            print(f"Error al listar documentos: {e}")
    elif choice == "4":
        try:
            manager = VectorSearchManager()
            structure = manager.get_bucket_structure()
            if 'error' not in structure:
                print(f"\n📁 Estructura del bucket: {structure['bucket_name']}")
                print("=" * 80)
                print(f"Total de archivos: {structure['total_files']}")
                print("\n📂 Carpetas encontradas:")
                for folder, info in structure['folders'].items():
                    print(f"\n  📁 {folder}/")
                    print(f"    - Archivos: {info['files']}")
                    print(f"    - Tamaño total: {info['size_bytes']:,} bytes")
                    print(f"    - Tipos: {', '.join(info['types'])}")
            else:
                print(f"Error al obtener estructura: {structure['error']}")
        except Exception as e:
            print(f"Error al mostrar estructura: {e}")
    elif choice == "5":
        print("Saliendo...")
    else:
        print("Opción inválida. Ejecutando consultas de prueba por defecto...")
        run_test_query()
