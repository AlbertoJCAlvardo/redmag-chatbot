"""
Script para verificar los esquemas actuales de las tablas de BigQuery.

Este script muestra la estructura actual de las tablas sin modificarlas.
"""

import os
import sys
from google.cloud import bigquery

# Agregar el directorio raíz al path para poder importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

def check_bigquery_schemas():
    """Verificar los esquemas actuales de las tablas de BigQuery."""
    client = bigquery.Client(project=config.project_id)
    
    # Lista de tablas a verificar
    tables = [
        config.bigquery_users_table,
        config.bigquery_messages_table,
        config.bigquery_context_table
    ]
    
    print(f"🔍 Verificando esquemas en proyecto: {config.project_id}")
    print("=" * 60)
    
    for table_id in tables:
        try:
            # Obtener información de la tabla
            table = client.get_table(table_id)
            
            print(f"\n📋 Tabla: {table_id}")
            print(f"   Descripción: {table.description or 'Sin descripción'}")
            print(f"   Filas: {table.num_rows:,}" if table.num_rows else "   Filas: 0")
            print(f"   Tamaño: {table.num_bytes / (1024*1024):.2f} MB")
            print(f"   Creada: {table.created}")
            print(f"   Modificada: {table.modified}")
            
            print("\n   📊 Esquema:")
            for field in table.schema:
                print(f"      - {field.name}: {field.field_type} {'(REQUIRED)' if field.mode == 'REQUIRED' else '(NULLABLE)'}")
                if field.description:
                    print(f"        Descripción: {field.description}")
            
            print("-" * 60)
            
        except Exception as e:
            print(f"\n❌ Error verificando tabla {table_id}: {e}")
            print("-" * 60)

def check_dataset_info():
    """Verificar información del dataset."""
    client = bigquery.Client(project=config.project_id)
    
    # Extraer dataset_id de la tabla de usuarios
    table_parts = config.bigquery_users_table.split('.')
    dataset_id = f"{table_parts[0]}.{table_parts[1]}"
    
    try:
        dataset = client.get_dataset(dataset_id)
        print(f"\n📁 Dataset: {dataset_id}")
        print(f"   Descripción: {dataset.description or 'Sin descripción'}")
        print(f"   Ubicación: {dataset.location}")
        print(f"   Creado: {dataset.created}")
        print(f"   Modificado: {dataset.modified}")
        
        # Listar todas las tablas en el dataset
        tables = list(client.list_tables(dataset_id))
        print(f"\n   📋 Tablas en el dataset ({len(tables)}):")
        for table in tables:
            print(f"      - {table.table_id}")
            
    except Exception as e:
        print(f"\n❌ Error verificando dataset {dataset_id}: {e}")

def main():
    """Función principal."""
    print("🔍 Verificador de Esquemas de BigQuery")
    print("=" * 60)
    
    check_dataset_info()
    check_bigquery_schemas()
    
    print("\n✅ Verificación completada")

if __name__ == "__main__":
    main() 