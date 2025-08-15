"""
Script para crear un nuevo índice de Vertex AI con Stream Update (768 dimensiones)
"""
from google.cloud import aiplatform
import time

def create_stream_index():
    # Configuración
    PROJECT_ID = "redmag-chatbot"
    LOCATION = "us-central1"
    
    # Inicializar Vertex AI
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    INDEX_NAME = "redmag_chatbot_el_chiludo"
    DIMENSIONS = 768
    
    print(f"🚀 Creando índice con STREAM UPDATE y {DIMENSIONS} dimensiones...")
    print(f"Proyecto: {PROJECT_ID}")
    print(f"Región: {LOCATION}")
    
    try:
        # Crear índice con Stream Update
        my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=INDEX_NAME,
            dimensions=DIMENSIONS,
            approximate_neighbors_count=5,
            distance_measure_type="DOT_PRODUCT_DISTANCE",
            index_update_method="STREAM_UPDATE",  # 🔥 ESTO ES LA CLAVE
            shard_size="SHARD_SIZE_SMALL",
            leaf_node_embedding_count=1000,
        )
        
        print("✅ Índice con Stream Update creado exitosamente!")
        print(f"ID completo: {my_index.resource_name}")
        
        index_id = my_index.resource_name.split('/')[-1]
        print(f"ID del índice: {index_id}")
        
        # Crear endpoint para Stream Update
        print("\n🔌 Creando endpoint para el índice...")
        endpoint = create_endpoint_for_stream_index(my_index, PROJECT_ID, LOCATION)
        
        if endpoint:
            print("\n" + "="*60)
            print("🔧 ACTUALIZA TU ARCHIVO DE CONFIGURACIÓN:")
            print("="*60)
            print(f"INDEX_ID = '{index_id}'")
            print(f"ENDPOINT_ID = '{endpoint.resource_name.split('/')[-1]}'")
            print(f"DEPLOYED_INDEX_ID = '{INDEX_NAME}'")
            print(f"LOCATION = 'us-central1'")
            print(f"PROJECT_ID = 'redmag-chatbot'")
            
            print("\n" + "="*60)
            print("🎯 VENTAJAS DEL STREAM UPDATE:")
            print("="*60)
            print("✅ Inserción inmediata (segundos, no horas)")
            print("✅ Sin timeouts largos")
            print("✅ Actualizaciones incrementales")
            print("✅ Perfecto para datos de APIs")
        
        return my_index, endpoint
        
    except Exception as e:
        print(f"❌ Error al crear el índice: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_endpoint_for_stream_index(index, project_id, location):
    """Crea un endpoint y despliega el índice (necesario para Stream Update)"""
    try:
        print("🔌 Creando endpoint...")
        
        # Crear endpoint
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=f"{index.display_name}-endpoint",
            public_endpoint_enabled=True  # Para acceso público
        )
        
        print(f"✅ Endpoint creado: {endpoint.resource_name}")
        
        # Desplegar índice al endpoint
        print("🚀 Desplegando índice al endpoint (puede tardar 15-20 minutos)...")
        
        deployed_index = endpoint.deploy_index(
            index=index,
            deployed_index_id=index.display_name,
            display_name=index.display_name,
            machine_type="e2-standard-2",  # Tipo de máquina mínimo
            min_replica_count=1,
            max_replica_count=1
        )
        
        print("✅ Índice desplegado exitosamente al endpoint!")
        return endpoint
        
    except Exception as e:
        print(f"❌ Error al crear endpoint: {e}")
        return None

def check_existing_indexes():
    """Verifica los índices existentes"""
    PROJECT_ID = "redmag-chatbot"
    LOCATION = "us-central1"
    
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    print("📋 Índices existentes:")
    try:
        indexes = aiplatform.MatchingEngineIndex.list()
        for i, index in enumerate(indexes, 1):
            print(f"  {i}. {index.display_name} (ID: {index.resource_name.split('/')[-1]})")
    except Exception as e:
        print(f"❌ Error al listar índices: {e}")

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 CREADOR DE ÍNDICE STREAM UPDATE - VERTEX AI")
    print("=" * 80)
    
    # Mostrar índices existentes
    check_existing_indexes()
    
    print("\n" + "🔥 CREANDO NUEVO ÍNDICE CON STREAM UPDATE...")
    index, endpoint = create_stream_index()
    
    if index and endpoint:
        print("\n🎉 ¡PROCESO COMPLETADO EXITOSAMENTE!")
        print("\n📝 PRÓXIMOS PASOS:")
        print("1. Actualiza tu config.py con los nuevos IDs")
        print("2. Ejecuta populate_vdb.py")
        print("3. ¡Los documentos se insertarán en segundos!")
    else:
        print("\n❌ Hubo un problema en la creación")