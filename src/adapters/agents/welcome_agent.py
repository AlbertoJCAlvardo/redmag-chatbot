"""
Welcome Agent for MentorIA Chatbot API.

Jarvis-style welcome agent that provides initial greeting and shows available options.
"""

import json
from typing import Dict, Any


class WelcomeAgent:
    """Jarvis-style welcome agent for initial user interaction."""

    def build_welcome_response(self, user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Build welcome response with Jarvis personality and available options.
        
        Args:
            user_profile: User profile data (optional)
            
        Returns:
            Welcome response with buttons and Jarvis personality
        """
        user_name = user_profile.get("nombre", "Docente") if user_profile else "Docente"
        
        welcome_message = f"""
¡Bienvenido, {user_name}! Soy MentorIA, su asistente educativo personal.

Como su fiel asistente, estoy aquí para ayudarle con todas sus necesidades educativas. 
He sido programado para ser su compañero en esta noble misión de educar.

¿En qué puedo asistirle hoy?
        """.strip()

        available_options = [
            {
                "label": "📚 Ayuda con Planeaciones",
                "value": "planeaciones",
                "description": "Crear y mejorar planeaciones didácticas"
            },
            {
                "label": "📖 Materiales Educativos (MEDs)",
                "value": "meds",
                "description": "Buscar y crear materiales educativos"
            },
            {
                "label": "🎯 Evaluación y Diagnóstico",
                "value": "evaluacion",
                "description": "Herramientas de evaluación y diagnóstico educativo"
            },
            {
                "label": "🔧 Metodologías de Enseñanza",
                "value": "metodologias",
                "description": "Estrategias y metodologías pedagógicas"
            },
            {
                "label": "📋 Programas Analíticos",
                "value": "programas",
                "description": "Ayuda con programas analíticos y secuencias"
            },
            {
                "label": "❓ Preguntas Generales",
                "value": "general",
                "description": "Consultas generales sobre educación"
            },
            {
                "label": "⚙️ Configurar Perfil",
                "value": "perfil",
                "description": "Actualizar información de su perfil"
            },
            {
                "label": "✍️ Escribir Consulta Personalizada",
                "value": "otro",
                "description": "Escribir su consulta específica directamente"
            }
        ]

        return {
            "type": "welcome",
            "data": {
                "message": welcome_message,
                "options": available_options,
                "personality": "jarvis",
                "show_typing": True
            }
        }

    def build_profile_setup_response(self) -> Dict[str, Any]:
        """
        Build response for profile setup with Jarvis personality.
        
        Returns:
            Profile setup response with questions
        """
        setup_message = """
Excelente elección, señor. Para brindarle la mejor asistencia posible, 
necesito conocer algunos detalles sobre su contexto educativo.

Permítame configurar su perfil para optimizar mis respuestas.
        """.strip()

        profile_questions = [
            {
                "field_name": "nivel",
                "question_text": "¿En qué nivel educativo enseña?",
                "options": [
                    {"label": "Preescolar", "value": "preescolar"},
                    {"label": "Primaria", "value": "primaria"},
                    {"label": "Secundaria", "value": "secundaria"},
                    {"label": "Preparatoria", "value": "preparatoria"},
                    {"label": "Universidad", "value": "universidad"}
                ]
            },
            {
                "field_name": "grado",
                "question_text": "¿Qué grado específico maneja?",
                "options": [
                    {"label": "1er Grado", "value": "primero"},
                    {"label": "2do Grado", "value": "segundo"},
                    {"label": "3er Grado", "value": "tercero"},
                    {"label": "4to Grado", "value": "cuarto"},
                    {"label": "5to Grado", "value": "quinto"},
                    {"label": "6to Grado", "value": "sexto"}
                ]
            },
            {
                "field_name": "materia",
                "question_text": "¿Qué materia o área enseña principalmente?",
                "options": [
                    {"label": "Matemáticas", "value": "matematicas"},
                    {"label": "Español", "value": "espanol"},
                    {"label": "Ciencias Naturales", "value": "ciencias"},
                    {"label": "Historia", "value": "historia"},
                    {"label": "Geografía", "value": "geografia"},
                    {"label": "Educación Física", "value": "edfisica"},
                    {"label": "Artes", "value": "artes"},
                    {"label": "Otra", "value": "otra"}
                ]
            }
        ]

        return {
            "type": "buttons",
            "data": {
                "message": setup_message,
                "questions": profile_questions,
                "personality": "jarvis"
            }
        }

    def build_custom_query_response(self) -> Dict[str, Any]:
        """
        Build response for custom query option with Jarvis personality.
        
        Returns:
            Custom query response asking user to write their question
        """
        custom_message = """
Excelente elección, señor. Permítame asistirle con su consulta personalizada.

Por favor, escriba su pregunta o solicitud específica y haré todo lo posible 
por brindarle la mejor asistencia posible.
        """.strip()

        return {
            "type": "text_input",
            "data": {
                "message": custom_message,
                "placeholder": "Escriba su consulta aquí...",
                "personality": "jarvis",
                "waiting_for_input": True
            }
        } 