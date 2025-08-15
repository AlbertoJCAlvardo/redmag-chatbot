"""
Script de prueba para el flujo completo del Chatbot Educativo con Gemini.

Este script demuestra:
- Identificación de intención con Gemini
- Respuestas preliminares inmediatas
- Recolección de información adicional
- Recomendación de contenido educativo
- Gestión de contexto y memoria de conversación
"""

import sys
import os
import logging
from typing import Dict, Any

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.gemini_service import GeminiService
from modules.config import config

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockConversationContext:
    """Contexto de conversación simulado para las pruebas."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.current_axh = None
        self.user_profile = {
            'nivel': None,
            'grado': None,
            'fase': None,
            'campo_formativo': None,
            'disciplina': None,
            'experiencia': None,
        }
        self.conversation_history = []
        self.awaiting_info = False
        self.info_needed = []
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Agrega un mensaje al historial."""
        message = {
            'role': role,
            'content': content,
            'metadata': metadata or {}
        }
        self.conversation_history.append(message)
    
    def get_context_summary(self) -> str:
        """Genera un resumen del contexto."""
        summary = f"Usuario: {self.user_id}\n"
        if self.current_axh:
            summary += f"Actividad actual: {self.current_axh}\n"
        
        profile_parts = []
        for key, value in self.user_profile.items():
            if value:
                profile_parts.append(f"{key}: {value}")
        
        if profile_parts:
            summary += f"Perfil: {', '.join(profile_parts)}\n"
        
        return summary


class MockVectorSearchManager:
    """Simulador del Vector Search Manager para las pruebas."""
    
    def __init__(self):
        self.mock_results = {
            'diagnóstico': [
                {
                    'id': 'doc_001',
                    'distance': '0.85',
                    'content': 'Guía completa para realizar diagnósticos iniciales en el aula. Incluye instrumentos de evaluación, rubricas y ejemplos prácticos.',
                    'metadata': {'type': 'MED', 'source': 'api_diagnostico'},
                    'type': 'MED'
                },
                {
                    'id': 'doc_002',
                    'distance': '0.78',
                    'content': 'Webinar: "Diagnóstico efectivo en el regreso a clases". Estrategias para conocer el nivel de tus estudiantes.',
                    'metadata': {'type': 'webinar', 'source': 'api_diagnostico'},
                    'type': 'webinar'
                }
            ],
            'planificación': [
                {
                    'id': 'doc_003',
                    'distance': '0.92',
                    'content': 'Programa Analítico paso a paso. Cómo estructurar tu plan de trabajo para el ciclo escolar.',
                    'metadata': {'type': 'guía', 'source': 'api_planificacion'},
                    'type': 'guía'
                }
            ]
        }
    
    def search_similar(self, query: str, num_neighbors: int = 5, include_content: bool = False):
        """Simula búsqueda vectorial."""
        # Determinar qué resultados devolver basado en el query
        if 'diagnóstico' in query.lower():
            return self.mock_results['diagnóstico']
        elif 'planificación' in query.lower() or 'programa' in query.lower():
            return self.mock_results['planificación']
        else:
            return []
    
    def get_document_by_id(self, doc_id: str):
        """Simula obtención de documento por ID."""
        # Buscar en todos los resultados mock
        for category in self.mock_results.values():
            for doc in category:
                if doc['id'] == doc_id:
                    return doc
        return None


def simulate_conversation():
    """Simula una conversación completa con el chatbot educativo."""
    
    try:
        print("🤖 CHATBOT EDUCATIVO - RED MAGISTERIAL")
        print("=" * 60)
        print("Simulando conversación con Gemini AI...")
        print("=" * 60)
        
        # Inicializar servicios
        gemini_service = GeminiService()
        vector_manager = MockVectorSearchManager()
        
        # Contexto de conversación
        user_id = "test_user_001"
        context = MockConversationContext(user_id)
        
        # Casos de prueba basados en el documento de requerimientos
        test_cases = [
            {
                'message': 'Busco crear un examen diagnóstico para mi grupo',
                'description': 'Caso de uso 1: Recomendación de contenido para diagnóstico'
            },
            {
                'message': 'Soy docente de nuevo ingreso y me asignaron 5to de primaria',
                'description': 'Información adicional del usuario'
            },
            {
                'message': '¿Qué recursos me recomiendas para la planificación?',
                'description': 'Caso de uso 2: Búsqueda de recursos de planificación'
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 CASO DE PRUEBA #{i}")
            print(f"Descripción: {test_case['description']}")
            print(f"Mensaje del usuario: '{test_case['message']}'")
            print("-" * 60)
            
            # Agregar mensaje al historial
            context.add_message('user', test_case['message'])
            
            # Procesar con Gemini
            user_profile = {k: v for k, v in context.user_profile.items() if v}
            conversation_context = {
                'current_axh': context.current_axh,
                'awaiting_info': context.awaiting_info,
                'info_needed': context.info_needed,
                'messages_count': len(context.conversation_history)
            }
            
            # Obtener intención y acciones de Gemini
            intent_result = gemini_service.get_intent_and_actions(
                test_case['message'], 
                user_profile, 
                conversation_context
            )
            
            if intent_result:
                print(f"🤖 Análisis de Gemini:")
                print(f"   Intención: {intent_result.get('intent', 'N/A')}")
                print(f"   Claves requeridas: {intent_result.get('required_personal_keys', [])}")
                print(f"   Claves de conversación: {intent_result.get('required_conversation_keys', [])}")
                
                # Procesar acciones
                actions = intent_result.get('actions', [])
                for action in actions:
                    action_type = action.get('type')
                    
                    if action_type == 'update_personal_profile':
                        # Actualizar perfil del usuario
                        new_fields = action.get('new_fields', {})
                        for key, value in new_fields.items():
                            if key in context.user_profile:
                                context.user_profile[key] = value
                        print(f"   ✅ Perfil actualizado: {new_fields}")
                    
                    elif action_type == 'vector_search':
                        # Simular búsqueda vectorial
                        query = action.get('query', '')
                        print(f"   🔍 Búsqueda vectorial: '{query}'")
                        
                        # Realizar búsqueda simulada
                        results = vector_manager.search_similar(query, num_neighbors=3, include_content=True)
                        
                        if results:
                            print(f"   📚 Resultados encontrados: {len(results)}")
                            
                            # Generar respuesta con Gemini
                            response = generate_recommendations_response(gemini_service, results, context)
                            print(f"   💬 Respuesta generada:")
                            print(f"      {response}")
                            
                            # Agregar respuesta al historial
                            context.add_message('assistant', response)
                        else:
                            print(f"   ❌ No se encontraron resultados")
                
                # Actualizar contexto
                if intent_result.get('intent') in ['diagnóstico', 'planificación', 'capacitación', 'evaluación', 'actividades', 'gestion']:
                    context.current_axh = intent_result.get('intent')
                
            else:
                print(f"   ❌ No se pudo procesar el mensaje")
            
            print("-" * 60)
            
            # Pausa entre casos
            if i < len(test_cases):
                input("Presiona Enter para continuar con el siguiente caso...")
        
        # Mostrar resumen final
        print(f"\n📋 RESUMEN FINAL DEL CONTEXTO")
        print(f"Usuario: {context.user_id}")
        print(f"AXH actual: {context.current_axh}")
        print(f"Perfil del usuario: {context.user_profile}")
        print(f"Mensajes en historial: {len(context.conversation_history)}")
        
        print("\n🎉 Simulación completada exitosamente!")
        
    except Exception as e:
        logger.error(f"Error durante la simulación: {e}", exc_info=True)
        print(f"❌ Error: {e}")


def generate_recommendations_response(gemini_service, results, context):
    """Genera una respuesta de recomendaciones usando Gemini."""
    
    try:
        # Preparar datos para el prompt
        axh_name = context.current_axh or "recursos educativos"
        user_profile = context.user_profile
        
        # Formatear resultados
        recs_text = ""
        for i, rec in enumerate(results[:3], 1):
            content_preview = rec['content'][:150] + "..." if len(rec['content']) > 150 else rec['content']
            recs_text += f"""
Recurso {i}:
- Tipo: {rec['type']}
- Relevancia: {rec['distance']}
- Contenido: {content_preview}
"""
        
        # Construir prompt para Gemini
        prompt = f"""
Eres un asistente educativo experto que ayuda a docentes mexicanos a encontrar recursos relevantes.

**Contexto del docente:**
- Nivel: {user_profile.get('nivel', 'No especificado')}
- Grado: {user_profile.get('grado', 'No especificado')}
- Experiencia: {user_profile.get('experiencia', 'No especificado')}
- Actividad que busca: {axh_name}

**Recursos encontrados:**
{recs_text}

**Tu tarea:**
Genera una respuesta que:
1. Sea cálida y empática
2. Reconozca el perfil específico del docente
3. Presente los recursos de forma atractiva y útil
4. Explique brevemente por qué cada recurso es relevante
5. Incluya emojis apropiados para hacer la respuesta más amigable
6. Termine invitando al docente a profundizar o buscar más recursos

**Formato deseado:**
- Máximo 3-4 párrafos
- Tono profesional pero cercano
- Incluir emojis educativos (📚, 🎯, 💡, etc.)
- Terminar con una pregunta que invite a la interacción

**Responde solo con la respuesta formateada, sin formato adicional.**
"""
        
        # Generar respuesta con Gemini
        response = gemini_service.model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error generando respuesta: {e}")
        return f"¡Perfecto! Encontré {len(results)} recursos relevantes para ti. ¿Te gustaría que profundice en alguno de ellos?"


def test_intent_detection():
    """Prueba específica de detección de intenciones."""
    
    try:
        gemini_service = GeminiService()
        
        print("\n🔍 PRUEBA DE DETECCIÓN DE INTENCIONES")
        print("=" * 50)
        
        test_messages = [
            "Necesito hacer un diagnóstico de mi grupo",
            "¿Cómo puedo planificar mis clases?",
            "Busco capacitación sobre la NEM",
            "Quiero evaluar a mis estudiantes",
            "Necesito actividades para matemáticas",
            "¿Cómo gestiono la documentación escolar?"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n📝 PRUEBA #{i}")
            print(f"Mensaje: '{message}'")
            print("-" * 40)
            
            # Simular contexto
            user_profile = {"nivel": "primaria", "grado": "5"}
            conversation_context = {"current_topic": "regreso a clases"}
            
            intent_result = gemini_service.get_intent_and_actions(message, user_profile, conversation_context)
            
            if intent_result:
                print(f"🤖 Intención: {intent_result.get('intent', 'N/A')}")
                print(f"📋 Acciones: {len(intent_result.get('actions', []))}")
                
                for action in intent_result.get('actions', []):
                    print(f"  - {action.get('type', 'N/A')}: {action.get('query', 'N/A')}")
            else:
                print(f"❌ No se pudo detectar la intención")
            
            # Pausa entre pruebas
            if i < len(test_messages):
                input("\nPresiona Enter para continuar...")
        
    except Exception as e:
        logger.error(f"Error en prueba de intenciones: {e}", exc_info=True)
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🧪 PRUEBAS DEL CHATBOT EDUCATIVO CON GEMINI")
    print("=" * 60)
    
    choice = input("Selecciona una opción:\n1. Simulación completa de conversación\n2. Prueba de detección de intenciones\n3. Ambas\nOpción (1-3): ").strip()
    
    if choice == "1":
        simulate_conversation()
    elif choice == "2":
        test_intent_detection()
    elif choice == "3":
        simulate_conversation()
        test_intent_detection()
    else:
        print("Opción inválida. Ejecutando simulación completa...")
        simulate_conversation() 