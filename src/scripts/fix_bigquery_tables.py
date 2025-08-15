"""
Script para corregir las definiciones de las tablas de BigQuery.

Este script actualiza las tablas para usar TIMESTAMP en lugar de DATETIME.
"""

import os
import sys
from google.cloud import bigquery

# Agregar el directorio raíz al path para poder importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

def fix_bigquery_tables():
    """Corregir las definiciones de las tablas de BigQuery."""
    client = bigquery.Client(project=config.project_id)
    
    # Definiciones corregidas de las tablas
    tables_definitions = {
        "users": """
        CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.users` (
            user_id STRING NOT NULL,
            created_at TIMESTAMP,
            last_usage TIMESTAMP,
            profile_data JSON
        )
        """,
        
        "messages": """
        CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.messages` (
            message_id STRING NOT NULL,
            conversation_id STRING,
            user_id STRING,
            timestamp TIMESTAMP,
            role STRING,
            content STRING,
            agent_response BOOLEAN
        )
        """,
        
        "conversation_context": """
        CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.conversation_context` (
            conversation_id STRING NOT NULL,
            user_id STRING,
            created_at TIMESTAMP,
            last_updated TIMESTAMP,
            context_data JSON,
            is_active BOOLEAN
        )
        """
    }
    
    # Extraer project_id y dataset_id de la tabla de usuarios
    table_parts = config.bigquery_users_table.split('.')
    project_id = table_parts[0]
    dataset_id = table_parts[1]
    
    print(f"Corrigiendo tablas en proyecto: {project_id}, dataset: {dataset_id}")
    
    for table_name, definition in tables_definitions.items():
        try:
            query = definition.format(project_id=project_id, dataset_id=dataset_id)
            job = client.query(query)
            job.result()  # Esperar a que termine
            print(f"✅ Tabla {table_name} corregida exitosamente")
        except Exception as e:
            print(f"❌ Error corrigiendo tabla {table_name}: {e}")

if __name__ == "__main__":
    fix_bigquery_tables() 