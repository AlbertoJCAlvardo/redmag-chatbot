#!/usr/bin/env python3
"""
Chatbot CLI - Interfaz de línea de comandos interactiva para el Chatbot Educativo.

Este script proporciona una interfaz conversacional donde puedes interactuar
directamente con el chatbot educativo de Red Magisterial.
"""

import sys
import os
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import colorama
from colorama import Fore, Style

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.gemini_service import GeminiService
from modules.vector_search import VectorSearchManager
from modules.config import config

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChatbotCLI:
    """Interfaz de línea de comandos para el chatbot educativo."""
    
    def __init__(self):
        self.gemini_service = None
        self.vector_manager = None
        self.user_profile = {
            'nivel': None,
            'grado': None,
            'fase': None,
            'campo_formativo': None,
            'disciplina': None,
            'experiencia': None,
        }
        self.conversation_history = []
        self.current_axh = None
        self.session_start = datetime.now()
        
        # Inicializar colorama para el efecto matrix
        colorama.init(autoreset=True)
        
    def initialize(self):
        """Inicializa los servicios de Gemini y Vector Search."""
        try:
            print("🔄 Inicializando Chatbot Educativo...")
            self.gemini_service = GeminiService()
            self.vector_manager = VectorSearchManager()
            print("✅ Chatbot inicializado correctamente!")
            return True
        except Exception as e:
            print(f"❌ Error al inicializar: {e}")
            return False
    
    def show_welcome(self):
        """Muestra el mensaje de bienvenida."""
        print("\n" + "="*60)
        print(Fore.GREEN + "CHATBOT EDUCATIVO - RED MAGISTERIAL")
        print("="*60)
        print(Fore.GREEN + "Hola! Soy tu asistente educativo especializado en ayudarte")
        print(Fore.GREEN + "durante la temporada de 'Regreso a Clases'.")
        print(Fore.GREEN + "\nComo docente, puedo ayudarte a ser mejor maestro con:")
        print(Fore.GREEN + "Diagnosticos iniciales para conocer a tu grupo")
        print(Fore.GREEN + "Planificacion y programas analiticos efectivos")
        print(Fore.GREEN + "Capacitacion sobre la Nueva Escuela Mexicana (NEM)")
        print(Fore.GREEN + "Evaluacion continua y formativa")
        print(Fore.GREEN + "Actividades y materiales didacticos")
        print(Fore.GREEN + "Gestion escolar y administrativa")
        print(Fore.GREEN + "\nEscribe 'salir' para terminar la conversacion.")
        print(Fore.GREEN + "Escribe 'ayuda' para ver comandos disponibles.")
        print(Fore.GREEN + "Escribe 'perfil' para ver tu informacion actual.")
        print("="*60)
    
    def show_help(self):
        """Muestra la ayuda del chatbot."""
        print(Fore.GREEN + "\nCOMANDOS DISPONIBLES:")
        print("-" * 30)
        print(Fore.GREEN + "salir          - Terminar la conversacion")
        print(Fore.GREEN + "ayuda          - Mostrar esta ayuda")
        print(Fore.GREEN + "perfil         - Ver tu perfil actual")
        print(Fore.GREEN + "limpiar        - Limpiar historial de conversacion")
        print(Fore.GREEN + "historial      - Ver historial de mensajes")
        print(Fore.GREEN + "estado         - Ver estado actual del chatbot")
        print(Fore.GREEN + "buscar <texto> - Buscar contenido especifico")
        print("-" * 30)
        print(Fore.GREEN + "Tambien puedes escribir cualquier consulta docente")
        print(Fore.GREEN + "y el chatbot te respondera automaticamente.")
    
    def show_profile(self):
        """Muestra el perfil actual del usuario."""
        print(Fore.GREEN + "\nTU PERFIL ACTUAL:")
        print("-" * 30)
        
        if not any(self.user_profile.values()):
            print(Fore.GREEN + "No tengo informacion sobre tu perfil docente.")
            print(Fore.GREEN + "Te ire preguntando para darte mejores recomendaciones.")
        else:
            for key, value in self.user_profile.items():
                if value:
                    display_name = {
                        'nivel': 'Nivel educativo',
                        'grado': 'Grado que impartes',
                        'fase': 'Fase de la NEM',
                        'campo_formativo': 'Campo formativo',
                        'disciplina': 'Asignatura/Disciplina',
                        'experiencia': 'Experiencia docente'
                    }.get(key, key)
                    print(Fore.GREEN + f"✓ {display_name}: {value}")
        
        if self.current_axh:
            print(Fore.GREEN + f"Actividad docente actual: {self.current_axh}")
        
        print(Fore.GREEN + f"Sesion iniciada: {self.session_start.strftime('%H:%M:%S')}")
        print(Fore.GREEN + f"Mensajes: {len(self.conversation_history)}")
    
    def show_history(self):
        """Muestra el historial de conversación."""
        if not self.conversation_history:
            print(Fore.GREEN + "\nNo hay mensajes en el historial.")
            return
        
        print(Fore.GREEN + f"\nHISTORIAL DE CONVERSACION ({len(self.conversation_history)} mensajes):")
        print("-" * 50)
        
        for i, msg in enumerate(self.conversation_history, 1):
            timestamp = msg.get('timestamp', 'N/A')
            role = "TU" if msg['role'] == 'user' else "BOT"
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(Fore.GREEN + f"{i:2d}. {role} [{timestamp}] {content}")
    
    def show_status(self):
        """Muestra el estado actual del chatbot."""
        print(Fore.GREEN + "\nESTADO DEL CHATBOT:")
        print("-" * 30)
        print(Fore.GREEN + f"Servicio Gemini: {'Conectado' if self.gemini_service else 'Desconectado'}")
        print(Fore.GREEN + f"Vector Search: {'Conectado' if self.vector_manager else 'Desconectado'}")
        print(Fore.GREEN + f"Docente: {self.get_user_summary()}")
        print(Fore.GREEN + f"Actividad docente: {self.current_axh or 'Ninguna'}")
        print(Fore.GREEN + f"Mensajes: {len(self.conversation_history)}")
        print(Fore.GREEN + f"Tiempo de sesion: {self.get_session_duration()}")
    
    def get_user_summary(self) -> str:
        """Genera un resumen del docente."""
        if not any(self.user_profile.values()):
            return "Docente sin información de perfil"
        
        parts = []
        if self.user_profile['nivel']:
            parts.append(self.user_profile['nivel'])
        if self.user_profile['grado']:
            parts.append(f"grado {self.user_profile['grado']}")
        if self.user_profile['experiencia']:
            parts.append(f"({self.user_profile['experiencia']})")
        
        return " ".join(parts) if parts else "Docente con información parcial"
    
    def get_session_duration(self) -> str:
        """Calcula la duración de la sesión."""
        duration = datetime.now() - self.session_start
        minutes = int(duration.total_seconds() // 60)
        seconds = int(duration.total_seconds() % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Agrega un mensaje al historial."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'metadata': metadata or {}
        }
        self.conversation_history.append(message)
    
    def process_message(self, user_message: str) -> str:
        """Procesa un mensaje del usuario y retorna la respuesta."""
        try:
            # Agregar mensaje del usuario al historial
            self.add_message('user', user_message)
            
            # Preparar contexto para Gemini
            user_profile = {k: v for k, v in self.user_profile.items() if v}
            conversation_context = {
                'current_axh': self.current_axh,
                'messages_count': len(self.conversation_history),
                'session_duration': self.get_session_duration()
            }
            
            # Obtener intención y acciones de Gemini
            intent_result = self.gemini_service.get_intent_and_actions(
                user_message, 
                user_profile, 
                conversation_context
            )
            
            if not intent_result:
                return "Lo siento, no pude entender tu consulta docente. Podrias reformularla?"
            
            # Procesar acciones
            response_parts = []
            
            # Actualizar perfil si es necesario
            actions = intent_result.get('actions', [])
            for action in actions:
                if action.get('type') == 'update_personal_profile':
                    new_fields = action.get('new_fields', {})
                    for key, value in new_fields.items():
                        if key in self.user_profile:
                            self.user_profile[key] = value
                    if new_fields:
                        response_parts.append(f"Actualice tu perfil docente con: {', '.join([f'{k}={v}' for k, v in new_fields.items()])}")
                
                elif action.get('type') == 'vector_search':
                    # Realizar búsqueda vectorial real
                    query = action.get('query', '')
                    print(f"🔍 Buscando: '{query}'")
                    
                    # Buscar en Vector Search
                    results = self.vector_manager.search_similar(query, num_neighbors=5, include_content=True)
                    
                    if results:
                        print(f"📚 Encontré {len(results)} recursos relevantes:")
                        
                        # Obtener contenido completo de los documentos
                        recommendations = []
                        for i, result in enumerate(results[:3], 1):  # Mostrar solo los 3 mejores
                            doc_id = result['id']
                            doc_details = self.vector_manager.get_document_by_id(doc_id)
                            
                            if doc_details and 'content' in doc_details:
                                recommendations.append({
                                    'id': doc_id,
                                    'distance': result['distance'],
                                    'content': doc_details['content'],
                                    'metadata': doc_details.get('metadata', {}),
                                    'type': doc_details.get('metadata', {}).get('type', 'documento')
                                })
                        
                        if recommendations:
                            # Generar respuesta con contenido real
                            response_parts.append(self.generate_recommendations_response(recommendations))
                        else:
                            response_parts.append("No encontre contenido especifico para tu practica docente, pero puedo ayudarte con informacion general.")
                    else:
                        response_parts.append("No encontre recursos especificos para tu consulta docente. Podrias ser mas especifico sobre lo que necesitas?")
            
            # Actualizar AXH actual
            intent = intent_result.get('intent')
            if intent in ['diagnóstico', 'planificación', 'capacitación', 'evaluación', 'actividades', 'gestion']:
                self.current_axh = intent
            
            # Generar respuesta principal si no hay búsqueda vectorial
            if not any(action.get('type') == 'vector_search' for action in actions):
                if intent == 'saludo':
                    response_parts.append("Hola! Soy tu asistente educativo. En que puedo ayudarte hoy en tu practica docente?")
                elif intent in ['diagnóstico', 'planificación', 'capacitación', 'evaluación', 'actividades', 'gestion']:
                    response_parts.append(self.generate_axh_response(intent, user_message))
                else:
                    response_parts.append("Entiendo tu consulta docente. Te ayudo a encontrar los recursos mas relevantes para mejorar tu practica educativa.")
            
            # Verificar si necesitamos más información y generar respuesta natural
            required_keys = intent_result.get('required_personal_keys', [])
            if required_keys:
                # Generar pregunta natural usando Gemini
                natural_question = self.generate_natural_question(required_keys, intent, user_message)
                if natural_question:
                    response_parts.append(f"\n{natural_question}")
            
            # Combinar respuesta
            full_response = " ".join(response_parts)
            
            # Agregar respuesta al historial
            self.add_message('assistant', full_response)
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error procesando mensaje: {e}")
            error_response = "Lo siento, tuve un problema procesando tu consulta docente. Podrias intentarlo de nuevo?"
            self.add_message('assistant', error_response)
            return error_response
    
    def generate_axh_response(self, axh: str, user_message: str) -> str:
        """Genera una respuesta específica para cada AXH enfocada en docentes."""
        axh_responses = {
            'diagnóstico': "Perfecto! Veo que necesitas ayuda con el diagnostico inicial. Esta es una parte fundamental del regreso a clases para conocer bien a tu grupo y adaptar tu enseñanza. Te ayudo a encontrar los mejores recursos para evaluar el nivel de tus estudiantes...",
            'planificación': "Excelente! La planificacion es clave para un regreso a clases exitoso. Te ayudo a encontrar los recursos mas utiles para organizar tu trabajo educativo y crear programas analiticos efectivos...",
            'capacitación': "Genial! La capacitacion es esencial para estar actualizado con la Nueva Escuela Mexicana. Te conecto con los mejores recursos de formacion docente para mejorar tu practica...",
            'evaluación': "Perfecto! La evaluacion es fundamental para el seguimiento del aprendizaje de tus estudiantes. Te ayudo a encontrar herramientas efectivas para evaluar de forma continua y formativa...",
            'actividades': "Excelente! Los recursos de actividades son clave para enriquecer tu practica docente y hacer tus clases mas dinamicas. Te muestro las mejores opciones para tu grupo...",
            'gestion': "Entiendo! La gestion escolar es importante para mantener todo organizado y cumplir con los requerimientos administrativos. Te ayudo con los recursos para optimizar tu trabajo..."
        }
        
        return axh_responses.get(axh, f"Perfecto! Te ayudo con {axh} para mejorar tu practica docente. Dejame buscar los mejores recursos para ti...")
    
    def generate_natural_question(self, required_keys: list, intent: str, user_message: str) -> str:
        """Genera una pregunta natural usando Gemini para obtener información adicional."""
        try:
            prompt = f"""
Eres un asistente educativo que necesita obtener informacion adicional de un docente para darle mejores recomendaciones.

Contexto:
- Intencion del docente: {intent}
- Mensaje original: "{user_message}"
- Informacion que necesitas: {', '.join(required_keys)}

Genera UNA pregunta natural y conversacional que:
1. Sea amigable y empatica
2. Explique brevemente por que necesitas esta informacion
3. Sea especifica para el contexto educativo mexicano
4. Use un tono profesional pero cercano
5. NO use emojis
6. Sea una sola pregunta, no una lista

Ejemplo de tono:
"Para ayudarte mejor con los recursos de diagnostico, podrias contarme en que nivel educativo trabajas?"

Responde solo con la pregunta, sin formato adicional.
"""
            
            response = self.gemini_service.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generando pregunta natural: {e}")
            # Fallback a pregunta básica
            return f"Para darte mejores recomendaciones, podrias contarme sobre {', '.join(required_keys)}?"
    
    def generate_recommendations_response(self, recommendations: list) -> str:
        """Genera una respuesta con las recomendaciones y contenido real de los documentos para docentes."""
        if not recommendations:
            return "No encontre contenido especifico para tu practica docente."
        
        response_parts = []
        response_parts.append("RECOMENDACIONES PARA TU PRACTICA DOCENTE")
        response_parts.append("=" * 60)
        
        for i, rec in enumerate(recommendations, 1):
            response_parts.append(f"\nRECURSO DOCENTE #{i}")
            response_parts.append(f"Tipo de material: {rec['type']}")
            response_parts.append(f"Relevancia para tu enseñanza: {rec['distance']}")
            
            # Mostrar contenido del documento
            content = rec['content']
            if len(content) > 400:
                content = content[:400] + "..."
            
            response_parts.append(f"CONTENIDO EDUCATIVO:")
            response_parts.append(f"{content}")
            
            # Mostrar metadatos si están disponibles
            metadata = rec.get('metadata', {})
            if metadata:
                response_parts.append(f"INFORMACION PARA DOCENTES:")
                for key, value in metadata.items():
                    if key != 'content' and value and key not in ['source', 'original_id']:
                        display_key = key.replace('_', ' ').title()
                        response_parts.append(f"   • {display_key}: {value}")
            
            response_parts.append("-" * 60)
        
        response_parts.append(f"\nRESUMEN: Encontre {len(recommendations)} recursos educativos relevantes para mejorar tu practica docente.")
        response_parts.append("Te gustaria que profundice en alguno de estos recursos o busco algo mas especifico para tu enseñanza?")
        
        return "\n".join(response_parts)
    
    def clear_history(self):
        """Limpia el historial de conversación."""
        self.conversation_history = []
        print(Fore.GREEN + "Historial de conversacion limpiado.")
    
    def run(self):
        """Ejecuta el chatbot CLI."""
        if not self.initialize():
            return
        
        self.show_welcome()
        
        while True:
            try:
                # Obtener input del usuario
                user_input = input("\n👤 Tú: ").strip()
                
                # Procesar comandos especiales
                if user_input.lower() == 'salir':
                    print("\n👋 ¡Gracias por usar el Chatbot Educativo! ¡Que tengas un excelente día!")
                    break
                elif user_input.lower() == 'ayuda':
                    self.show_help()
                    continue
                elif user_input.lower() == 'perfil':
                    self.show_profile()
                    continue
                elif user_input.lower() == 'historial':
                    self.show_history()
                    continue
                elif user_input.lower() == 'estado':
                    self.show_status()
                    continue
                elif user_input.lower() == 'limpiar':
                    self.clear_history()
                    continue
                elif user_input.lower().startswith('buscar '):
                    query = user_input[7:].strip()  # Remover "buscar " del inicio
                    if query:
                        print(Fore.GREEN + f"Buscando: '{query}'")
                        results = self.vector_manager.search_similar(query, num_neighbors=5, include_content=True)
                        if results:
                            print(Fore.GREEN + f"Encontre {len(results)} resultados:")
                            for i, result in enumerate(results[:3], 1):
                                print(Fore.GREEN + f"\nResultado #{i}:")
                                print(Fore.GREEN + f"   ID: {result['id']}")
                                print(Fore.GREEN + f"   Relevancia: {result['distance']}")
                                if 'metadata' in result:
                                    print(Fore.GREEN + f"   Metadatos: {result['metadata']}")
                        else:
                            print(Fore.GREEN + "No se encontraron resultados.")
                    else:
                        print(Fore.GREEN + "Uso: buscar <texto a buscar>")
                    continue
                elif not user_input:
                    continue
                
                # Procesar mensaje normal
                print(Fore.GREEN + "Procesando...")
                response = self.process_message(user_input)
                print(Fore.GREEN + f"Asistente: {response}")
                
            except KeyboardInterrupt:
                print(Fore.GREEN + "\n\nHasta luego! Que tengas un excelente dia!")
                break
            except Exception as e:
                logger.error(f"Error en el loop principal: {e}")
                print(Fore.GREEN + f"Error: {e}")
                print(Fore.GREEN + "Intenta escribir tu mensaje de nuevo.")


def main():
    """Función principal."""
    print(Fore.GREEN + "Iniciando Chatbot CLI...")
    
    # Verificar configuración
    if not config.gemini_api_key:
        print(Fore.GREEN + "Error: GEMINI_API_KEY no esta configurado en el archivo .env")
        print(Fore.GREEN + "Por favor, agrega tu API key de Gemini al archivo .env")
        return
    
    # Crear y ejecutar el chatbot
    chatbot = ChatbotCLI()
    chatbot.run()


if __name__ == "__main__":
    main() 