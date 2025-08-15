"""
Script para probar el sistema de bienvenida con personalidad Jarvis.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_welcome_message():
    """Probar el mensaje de bienvenida inicial."""
    print("🤖 Probando mensaje de bienvenida con personalidad Jarvis")
    print("=" * 60)
    
    user_id = "test_user_jarvis_123"
    
    # Primera interacción (debería mostrar bienvenida)
    payload = {
        "user_id": user_id,
        "message": "Hola"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Respuesta recibida: {response.status_code}")
            print(f"📋 Tipo de respuesta: {data.get('response_type')}")
            print(f"🆔 Conversation ID: {data.get('conversation_id')}")
            
            # Mostrar el mensaje de bienvenida
            if data.get('response_type') == 'welcome':
                welcome_data = data.get('data', {})
                print(f"\n🤖 Mensaje de Jarvis:")
                print(f"   {welcome_data.get('message', 'Sin mensaje')}")
                
                # Mostrar opciones disponibles
                options = welcome_data.get('options', [])
                print(f"\n📋 Opciones disponibles ({len(options)}):")
                for i, option in enumerate(options, 1):
                    print(f"   {i}. {option.get('label')} - {option.get('description')}")
                
                return data.get('conversation_id'), options
            else:
                print(f"❌ No se recibió mensaje de bienvenida")
                print(f"   Respuesta: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return None, []
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
            return None, []
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return None, []

def test_menu_selection(conversation_id, options):
    """Probar la selección de opciones del menú."""
    print(f"\n🎯 Probando selección de opciones del menú")
    print("=" * 60)
    
    user_id = "test_user_jarvis_123"
    
    # Probar selección de perfil
    print("📝 Probando selección de 'Configurar Perfil'...")
    payload = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "user_data": [
            {"field": "menu_option", "value": "perfil"}
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Respuesta recibida: {response.status_code}")
            print(f"📋 Tipo de respuesta: {data.get('response_type')}")
            
            if data.get('response_type') == 'buttons':
                button_data = data.get('data', {})
                print(f"\n🤖 Respuesta de Jarvis:")
                print(f"   {button_data.get('message', 'Sin mensaje')}")
                
                # Mostrar preguntas de perfil
                questions = button_data.get('questions', [])
                print(f"\n❓ Preguntas de perfil ({len(questions)}):")
                for i, question in enumerate(questions, 1):
                    print(f"   {i}. {question.get('question_text')}")
                    options = question.get('options', [])
                    for j, option in enumerate(options, 1):
                        print(f"      {j}. {option.get('label')}")
                
                return True
            else:
                print(f"❌ No se recibió respuesta de botones")
                print(f"   Respuesta: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return False
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False

def test_other_menu_options(conversation_id, options):
    """Probar otras opciones del menú."""
    print(f"\n🔧 Probando otras opciones del menú")
    print("=" * 60)
    
    user_id = "test_user_jarvis_123"
    
    # Probar algunas opciones del menú
    test_options = ["planeaciones", "meds", "evaluacion", "otro"]
    
    for option in test_options:
        print(f"📝 Probando opción: {option}")
        
        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "user_data": [
                {"field": "menu_option", "value": option}
            ]
        }
        
        try:
            response = requests.post(f"{BASE_URL}/chat", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Respuesta: {data.get('response_type')}")
                
                # Mostrar parte del mensaje si es texto
                if data.get('response_type') == 'text':
                    text_data = data.get('data', {})
                    message = text_data.get('text', '')[:100] + "..." if len(text_data.get('text', '')) > 100 else text_data.get('text', '')
                    print(f"   💬 {message}")
                    
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        time.sleep(0.5)  # Pequeña pausa entre pruebas

def test_custom_query_option(conversation_id):
    """Probar específicamente la opción de consulta personalizada."""
    print(f"\n✍️ Probando opción de consulta personalizada")
    print("=" * 60)
    
    user_id = "test_user_jarvis_123"
    
    # Probar selección de "otro"
    print("📝 Probando selección de 'Escribir Consulta Personalizada'...")
    payload = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "user_data": [
            {"field": "menu_option", "value": "otro"}
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Respuesta recibida: {response.status_code}")
            print(f"📋 Tipo de respuesta: {data.get('response_type')}")
            
            if data.get('response_type') == 'text_input':
                input_data = data.get('data', {})
                print(f"\n🤖 Respuesta de Jarvis:")
                print(f"   {input_data.get('message', 'Sin mensaje')}")
                print(f"   Placeholder: {input_data.get('placeholder', 'Sin placeholder')}")
                print(f"   Esperando input: {input_data.get('waiting_for_input', False)}")
                
                # Simular una consulta personalizada
                print(f"\n📝 Simulando consulta personalizada...")
                custom_payload = {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "message": "Necesito ayuda para crear una actividad de matemáticas para niños de 3er grado sobre fracciones"
                }
                
                custom_response = requests.post(f"{BASE_URL}/chat", json=custom_payload)
                
                if custom_response.status_code == 200:
                    custom_data = custom_response.json()
                    print(f"✅ Consulta personalizada procesada")
                    print(f"📋 Tipo de respuesta: {custom_data.get('response_type')}")
                    
                    # Mostrar parte del mensaje si es texto
                    if custom_data.get('response_type') == 'text':
                        text_data = custom_data.get('data', {})
                        message = text_data.get('text', '')[:150] + "..." if len(text_data.get('text', '')) > 150 else text_data.get('text', '')
                        print(f"💬 Respuesta: {message}")
                        
                else:
                    print(f"❌ Error en consulta personalizada: {custom_response.status_code}")
                    
                return True
            else:
                print(f"❌ No se recibió respuesta de text_input")
                print(f"   Respuesta: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return False
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False

def test_profile_setup():
    """Probar la configuración completa del perfil."""
    print(f"\n⚙️ Probando configuración completa del perfil")
    print("=" * 60)
    
    user_id = "test_user_profile_456"
    
    # Primera interacción para obtener bienvenida
    payload1 = {
        "user_id": user_id,
        "message": "Hola"
    }
    
    try:
        response1 = requests.post(f"{BASE_URL}/chat", json=payload1)
        
        if response1.status_code == 200:
            data1 = response1.json()
            conversation_id = data1.get('conversation_id')
            
            # Seleccionar configuración de perfil
            payload2 = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "user_data": [
                    {"field": "menu_option", "value": "perfil"}
                ]
            }
            
            response2 = requests.post(f"{BASE_URL}/chat", json=payload2)
            
            if response2.status_code == 200:
                # Configurar perfil completo
                payload3 = {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "user_data": [
                        {"field": "nivel", "value": "primaria"},
                        {"field": "grado", "value": "quinto"},
                        {"field": "materia", "value": "matematicas"},
                        {"field": "nombre", "value": "Profesor García"}
                    ]
                }
                
                response3 = requests.post(f"{BASE_URL}/chat", json=payload3)
                
                if response3.status_code == 200:
                    data3 = response3.json()
                    print(f"✅ Perfil configurado exitosamente")
                    print(f"📋 Respuesta: {data3.get('response_type')}")
                    
                    # Probar que el perfil se guardó
                    payload4 = {
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "message": "¿Cuál es mi perfil?"
                    }
                    
                    response4 = requests.post(f"{BASE_URL}/chat", json=payload4)
                    
                    if response4.status_code == 200:
                        data4 = response4.json()
                        print(f"✅ Consulta de perfil exitosa")
                        print(f"📋 Respuesta: {data4.get('response_type')}")
                        
                else:
                    print(f"❌ Error configurando perfil: {response3.status_code}")
                    
            else:
                print(f"❌ Error seleccionando perfil: {response2.status_code}")
                
        else:
            print(f"❌ Error en bienvenida: {response1.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Función principal."""
    print("🚀 Iniciando pruebas del sistema de bienvenida Jarvis")
    
    # Esperar que el servidor esté listo
    print("⏳ Esperando que el servidor esté listo...")
    time.sleep(2)
    
    # Probar mensaje de bienvenida
    conversation_id, options = test_welcome_message()
    
    if conversation_id and options:
        # Probar selección de menú
        test_menu_selection(conversation_id, options)
        
        # Probar otras opciones
        test_other_menu_options(conversation_id, options)
        
        # Probar opción de consulta personalizada
        test_custom_query_option(conversation_id)
        
        # Probar configuración de perfil
        test_profile_setup()
    
    print("\n" + "=" * 60)
    print("🏁 Pruebas del sistema de bienvenida completadas")

if __name__ == "__main__":
    main() 