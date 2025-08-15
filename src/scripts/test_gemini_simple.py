"""
Script de prueba simple para el servicio de Gemini.
"""

import sys
import os
import logging

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.gemini_service import GeminiService
from modules.config import config

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_gemini_basic():
    """Prueba básica del servicio de Gemini."""
    
    try:
        print("🧪 PRUEBA BÁSICA DE GEMINI SERVICE")
        print("=" * 50)
        
        # Verificar configuración
        if not config.gemini_api_key:
            print("❌ Error: GEMINI_API_KEY no está configurado en el archivo .env")
            return
        
        print(f"✅ API Key configurada: {config.gemini_api_key[:10]}...")
        
        # Inicializar servicio
        print("🔄 Inicializando GeminiService...")
        gemini_service = GeminiService()
        print("✅ GeminiService inicializado correctamente")
        
        # Probar detección de intención
        print("\n📝 Probando detección de intención...")
        
        test_messages = [
            "Busco crear un examen diagnóstico para mi grupo",
            "¿Cómo puedo planificar mis clases?",
            "Necesito capacitación sobre la NEM"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n--- PRUEBA #{i} ---")
            print(f"Mensaje: '{message}'")
            
            try:
                # Simular perfil de usuario
                user_profile = {
                    "nivel": "primaria",
                    "grado": "5",
                    "experiencia": "nuevo ingreso"
                }
                
                # Simular contexto de conversación
                conversation_context = {
                    "current_topic": "regreso a clases",
                    "messages_count": 1
                }
                
                # Llamar a Gemini
                result = gemini_service.get_intent_and_actions(
                    message, 
                    user_profile, 
                    conversation_context
                )
                
                if result:
                    print(f"✅ Intención detectada: {result.get('intent', 'N/A')}")
                    print(f"📋 Acciones: {len(result.get('actions', []))}")
                    print(f"🔑 Claves requeridas: {result.get('required_personal_keys', [])}")
                    
                    # Mostrar acciones específicas
                    for action in result.get('actions', []):
                        print(f"  - {action.get('type', 'N/A')}: {action.get('query', 'N/A')}")
                else:
                    print("❌ No se pudo detectar la intención")
                    
            except Exception as e:
                print(f"❌ Error en prueba #{i}: {e}")
        
        print("\n🎉 Prueba completada exitosamente!")
        
    except Exception as e:
        logger.error(f"Error en prueba básica: {e}", exc_info=True)
        print(f"❌ Error general: {e}")


def test_gemini_prompts():
    """Prueba la generación de prompts."""
    
    try:
        print("\n🧪 PRUEBA DE GENERACIÓN DE PROMPTS")
        print("=" * 50)
        
        gemini_service = GeminiService()
        
        # Probar construcción de prompt
        user_message = "Necesito ayuda con el diagnóstico de mi grupo"
        user_profile = {"nivel": "primaria", "grado": "3"}
        conversation_context = {"current_topic": "diagnóstico"}
        
        prompt = gemini_service._build_intent_prompt(user_message, user_profile, conversation_context)
        
        print("📝 Prompt generado:")
        print("-" * 30)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("-" * 30)
        
        print("✅ Prompt generado correctamente")
        
    except Exception as e:
        logger.error(f"Error en prueba de prompts: {e}", exc_info=True)
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🚀 INICIANDO PRUEBAS DE GEMINI SERVICE")
    print("=" * 60)
    
    # Ejecutar pruebas
    test_gemini_basic()
    test_gemini_prompts()
    
    print("\n🏁 TODAS LAS PRUEBAS COMPLETADAS") 