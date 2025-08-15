"""
Script de prueba para el flujo completo del Chatbot con BigQuery y Gemini.
"""

import logging
import uuid
import os
import sys

# Añadir la ruta del proyecto al path para que encuentre los módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.bigquery_manager import BigQueryManager
from modules.vector_search import VectorSearchManager
from modules.gemini_service import GeminiService
from modules.chat_logic import ChatManager

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
USER_ID = "test-user-12345"
CONVERSATION_ID = f"test-conv-{uuid.uuid4()}"

def run_test():
    """Ejecuta una simulación de conversación para probar la lógica del chatbot."""
    print("--- INICIANDO PRUEBA DE SIMULACIÓN DE CHATBOT ---")

    try:
        # 1. Inicializar todos los servicios
        bq_manager = BigQueryManager()
        vector_manager = VectorSearchManager()
        gemini_service = GeminiService()
        chat_manager = ChatManager(bq_manager, vector_manager, gemini_service)
        print("✅ Servicios inicializados correctamente.")

        # 2. Simular primera interacción del usuario
        print("\n--- TURNO 1: El usuario pide ayuda ---")
        user_message_1 = "Hola, necesito ayuda para hacer mi programa analítico"
        print(f"👤 Usuario: {user_message_1}")

        response_1 = chat_manager.handle_message(USER_ID, CONVERSATION_ID, user_message_1)
        print(f"🤖 Asistente: {response_1['assistant_message']}")
        
        # Verificar el estado de la memoria
        context_after_turn_1 = bq_manager.get_conversation_context(CONVERSATION_ID)
        print(f"📝 Contexto guardado: {context_after_turn_1}")

        # 3. Simular segunda interacción del usuario (respondiendo a la pregunta del bot)
        print("\n--- TURNO 2: El usuario proporciona más información ---")
        user_message_2 = "Soy de 5to de primaria"
        print(f"👤 Usuario: {user_message_2}")

        response_2 = chat_manager.handle_message(USER_ID, CONVERSATION_ID, user_message_2)
        print(f"🤖 Asistente: {response_2['assistant_message']}")

        # Verificar cómo se actualizó la memoria
        profile_after_turn_2 = bq_manager.get_user_profile(USER_ID)
        print(f"🧠 Perfil de usuario actualizado: {profile_after_turn_2}")
        context_after_turn_2 = bq_manager.get_conversation_context(CONVERSATION_ID)
        print(f"📝 Contexto actualizado: {context_after_turn_2}")

        print("\n--- PRUEBA FINALIZADA EXITOSAMENTE ---")

    except Exception as e:
        logging.error("La prueba falló.", exc_info=True)
        print(f"\n--- ❌ LA PRUEBA FALLÓ: {e} ---")

if __name__ == "__main__":
    run_test()
