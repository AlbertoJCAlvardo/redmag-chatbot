"""
Script para probar el flujo completo del contexto de conversación.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_context_flow():
    """Probar el flujo completo del contexto."""
    print("🔄 Probando flujo completo del contexto de conversación")
    print("=" * 60)
    
    user_id = "context_test_user_123"
    
    # Paso 1: Primera interacción (bienvenida)
    print("1️⃣ Primera interacción (bienvenida)...")
    payload1 = {
        "user_id": user_id,
        "message": "Hola"
    }
    
    response1 = requests.post(f"{BASE_URL}/chat", json=payload1)
    data1 = response1.json()
    
    print(f"   Status: {response1.status_code}")
    print(f"   Tipo: {data1.get('response_type')}")
    print(f"   Conversation ID: {data1.get('conversation_id')}")
    
    conversation_id = data1.get('conversation_id')
    
    # Paso 2: Seleccionar "perfil"
    print("\n2️⃣ Seleccionando 'perfil'...")
    payload2 = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "user_data": [
            {"field": "menu_option", "value": "perfil"}
        ]
    }
    
    response2 = requests.post(f"{BASE_URL}/chat", json=payload2)
    data2 = response2.json()
    
    print(f"   Status: {response2.status_code}")
    print(f"   Tipo: {data2.get('response_type')}")
    
    if data2.get('response_type') == 'buttons':
        print("   ✅ ¡Éxito! Se recibió respuesta de botones")
        button_data = data2.get('data', {})
        print(f"   Mensaje: {button_data.get('message', 'Sin mensaje')[:100]}...")
    else:
        print("   ❌ No se recibió respuesta de botones")
        print(f"   Respuesta: {data2.get('response_type')}")
    
    # Paso 3: Configurar perfil
    print("\n3️⃣ Configurando perfil...")
    payload3 = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "user_data": [
            {"field": "nivel", "value": "primaria"},
            {"field": "grado", "value": "quinto"},
            {"field": "materia", "value": "matematicas"}
        ]
    }
    
    response3 = requests.post(f"{BASE_URL}/chat", json=payload3)
    data3 = response3.json()
    
    print(f"   Status: {response3.status_code}")
    print(f"   Tipo: {data3.get('response_type')}")
    
    # Paso 4: Enviar mensaje de texto
    print("\n4️⃣ Enviando mensaje de texto...")
    payload4 = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "message": "Necesito ayuda con fracciones"
    }
    
    response4 = requests.post(f"{BASE_URL}/chat", json=payload4)
    data4 = response4.json()
    
    print(f"   Status: {response4.status_code}")
    print(f"   Tipo: {data4.get('response_type')}")
    
    if data4.get('response_type') == 'text':
        text_data = data4.get('data', {})
        print(f"   ✅ Respuesta de texto recibida")
        print(f"   Mensaje: {text_data.get('text', 'Sin mensaje')[:150]}...")
    else:
        print(f"   ❌ No se recibió respuesta de texto")
        print(f"   Respuesta: {data4.get('response_type')}")
    
    # Paso 5: Seleccionar "otro"
    print("\n5️⃣ Seleccionando 'otro'...")
    payload5 = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "user_data": [
            {"field": "menu_option", "value": "otro"}
        ]
    }
    
    response5 = requests.post(f"{BASE_URL}/chat", json=payload5)
    data5 = response5.json()
    
    print(f"   Status: {response5.status_code}")
    print(f"   Tipo: {data5.get('response_type')}")
    
    if data5.get('response_type') == 'text_input':
        print("   ✅ ¡Éxito! Se recibió respuesta de text_input")
        input_data = data5.get('data', {})
        print(f"   Mensaje: {input_data.get('message', 'Sin mensaje')[:100]}...")
    else:
        print("   ❌ No se recibió respuesta de text_input")
        print(f"   Respuesta: {data5.get('response_type')}")
    
    print("\n" + "=" * 60)
    print("🏁 Prueba del flujo de contexto completada")

if __name__ == "__main__":
    test_context_flow() 