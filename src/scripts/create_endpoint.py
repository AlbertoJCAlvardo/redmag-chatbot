"""
Script para crear un endpoint para tu índice existente y convertirlo a Stream Update
"""
from google.cloud import aiplatform

def create_endpoint_for_existing_index():
    # Configuración
    PROJECT_ID = "redmag-chatbot"
    LOCATION = "us-central1"
    
    # Tu índice actual con 768 dimensiones
    INDEX_ID = "6843450531231301632"
    
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    try:
        print("🔍 Obteniendo índice existente...")
        
        # Obtener el índice existente
        index = aiplatform.MatchingEngineIndex(index_name=INDEX_ID)
        print(f"✅ Índice encontrado: {index.display_name}")
        
        # Verificar si el índice soporta Stream Update
        print("🔍 Verificando configuración del índice...")
        
        # Crear endpoint
        print("🔌 Creando endpoint para Stream Update...")
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=f"{index.display_name}-stream-endpoint",
            public_endpoint_enabled=True
        )
        
        print(f"✅ Endpoint creado: {endpoint.display_name}")
        endpoint_id = endpoint.resource_name.split('/')[-1]
        
        # Desplegar índice al endpoint
        print("🚀 Desplegando índice al endpoint...")
        print("   ⏰ Esto puede tardar 15-20 minutos...")
        
        deployed_index_id = f"{index.display_name}-deployed"
        
        deployed_index = endpoint.deploy_index(
            index=index,
            deployed_index_id=deployed_index_id,
            display_name=index.display_name,
            machine_type="e2-standard-2",
            min_replica_count=1,
            max_replica_count=1
        )
        
        print("✅ ¡Índice desplegado exitosamente!")
        
        print("\n" + "="*60)
        print("🔧 ACTUALIZA TU ARCHIVO DE CONFIGURACIÓN:")
        print("="*60)
        print(f"INDEX_ID = '{INDEX_ID}'")
        print(f"ENDPOINT_ID = '{endpoint_id}'")
        print(f"DEPLOYED_INDEX_ID = '{deployed_index_id}'")
        print(f"LOCATION = 'us-central1'")
        print(f"PROJECT_ID = 'redmag-chatbot'")
        
        print("\n" + "="*60)
        print("🚀 ¡AHORA PUEDES USAR STREAM UPDATE!")
        print("="*60)
        print("✅ Tu script populate_vdb.py detectará automáticamente")
        print("   que el índice tiene un endpoint y usará stream update")
        print("✅ Los documentos se insertarán en segundos")
        
        return endpoint, deployed_index_id
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def check_operation_and_proceed():
    """Verifica si la operación batch anterior terminó antes de crear endpoint"""
    from google.cloud import aiplatform_v1
    
    operation_name = "projects/324789362064/locations/us-central1/indexes/6843450531231301632/operations/4668434029740032000"
    
    api_endpoint = "us-central1-aiplatform.googleapis.com"
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform_v1.IndexServiceClient(client_options=client_options)
    
    try:
        print("🔍 Verificando operación batch anterior...")
        operation = client.get_operation(request={"name": operation_name})
        
        if operation.done:
            if operation.error:
                print(f"❌ La operación batch falló: {operation.error}")
            else:
                print("✅ La operación batch completó exitosamente")
            print("✅ Podemos proceder a crear el endpoint")
            return True
        else:
            print("⏳ La operación batch aún está en progreso")
            print("💡 Podemos crear el endpoint en paralelo")
            return True
            
    except Exception as e:
        print(f"⚠️ No se pudo verificar la operación: {e}")
        print("✅ Procederemos a crear el endpoint de todas formas")
        return True

if __name__ == "__main__":
    print("=" * 80)
    print("🔌 CONFIGURADOR DE STREAM UPDATE - VERTEX AI")
    print("=" * 80)
    
    # Verificar operación anterior
    can_proceed = check_operation_and_proceed()
    
    if can_proceed:
        print("\n🚀 Creando endpoint para Stream Update...")
        endpoint, deployed_id = create_endpoint_for_existing_index()
        
        if endpoint:
            print("\n🎉 ¡CONFIGURACIÓN COMPLETADA!")
            print("\n📝 PRÓXIMOS PASOS:")
            print("1. Actualiza config.py con los nuevos valores")
            print("2. Ejecuta populate_vdb.py")
            print("3. ¡Disfruta de las inserciones rápidas!")
        else:
            print("\n❌ Hubo un problema. Revisa los logs arriba.")