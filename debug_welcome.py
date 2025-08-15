"""
Script simple para debuggear el sistema de bienvenida.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_simple_flow():
    """Probar un flujo simple paso a paso."""
    print("🔍 Debug del sistema de bienvenida")
    print("=" * 50)
    
    user_id = "debug_user_123"
    
    # Paso 1: Primera interacción (debería mostrar bienvenida)
    print("1️⃣ Primera interacción...")
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
    print(f"   Conversation ID: {data2.get('conversation_id')}")
    
    if data2.get('response_type') == 'buttons':
        print("   ✅ ¡Éxito! Se recibió respuesta de botones")
        button_data = data2.get('data', {})
        print(f"   Mensaje: {button_data.get('message', 'Sin mensaje')[:100]}...")
    else:
        print("   ❌ No se recibió respuesta de botones")
        print(f"   Respuesta completa: {json.dumps(data2, indent=2, ensure_ascii=False)}")
    
    # Paso 3: Seleccionar "otro"
    print("\n3️⃣ Seleccionando 'otro'...")
    payload3 = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "user_data": [
            {"field": "menu_option", "value": "otro"}
        ]
    }
    
    response3 = requests.post(f"{BASE_URL}/chat", json=payload3)
    data3 = response3.json()
    
    print(f"   Status: {response3.status_code}")
    print(f"   Tipo: {data3.get('response_type')}")
    print(f"   Conversation ID: {data3.get('conversation_id')}")
    
    if data3.get('response_type') == 'text_input':
        print("   ✅ ¡Éxito! Se recibió respuesta de text_input")
        input_data = data3.get('data', {})
        print(f"   Mensaje: {input_data.get('message', 'Sin mensaje')[:100]}...")
    else:
        print("   ❌ No se recibió respuesta de text_input")
        print(f"   Respuesta completa: {json.dumps(data3, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    test_simple_flow() 