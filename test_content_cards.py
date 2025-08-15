"""
Script para probar específicamente la generación de content_cards.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_content_cards():
    """Probar la generación de content_cards."""
    print("📚 Probando generación de content_cards")
    print("=" * 60)
    
    user_id = "content_test_user_456"
    
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
    
    # Paso 2: Seleccionar "planeaciones"
    print("\n2️⃣ Seleccionando 'planeaciones'...")
    payload2 = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "user_data": [
            {"field": "menu_option", "value": "planeaciones"}
        ]
    }
    
    response2 = requests.post(f"{BASE_URL}/chat", json=payload2)
    data2 = response2.json()
    
    print(f"   Status: {response2.status_code}")
    print(f"   Tipo: {data2.get('response_type')}")
    
    if data2.get('response_type') == 'content_cards':
        print("   ✅ ¡Éxito! Se recibieron content_cards")
        content_data = data2.get('data', {})
        print(f"   Intro text: {content_data.get('intro_text', 'Sin intro')}")
        print(f"   Total results: {content_data.get('total_results', 0)}")
        
        content_cards = content_data.get('content_cards', [])
        print(f"   Content cards: {len(content_cards)}")
        
        for i, card in enumerate(content_cards[:3], 1):  # Mostrar solo los primeros 3
            print(f"   Card {i}:")
            print(f"     ID: {card.get('id', 'Sin ID')}")
            print(f"     Type: {card.get('content_type', 'Sin tipo')}")
            print(f"     Title: {card.get('title', 'Sin título')[:50]}...")
    else:
        print(f"   ❌ No se recibieron content_cards")
        print(f"   Respuesta: {data2.get('response_type')}")
    
    # Paso 3: Seleccionar "meds"
    print("\n3️⃣ Seleccionando 'meds'...")
    payload3 = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "user_data": [
            {"field": "menu_option", "value": "meds"}
        ]
    }
    
    response3 = requests.post(f"{BASE_URL}/chat", json=payload3)
    data3 = response3.json()
    
    print(f"   Status: {response3.status_code}")
    print(f"   Tipo: {data3.get('response_type')}")
    
    if data3.get('response_type') == 'content_cards':
        print("   ✅ ¡Éxito! Se recibieron content_cards")
        content_data = data3.get('data', {})
        print(f"   Intro text: {content_data.get('intro_text', 'Sin intro')}")
        print(f"   Total results: {content_data.get('total_results', 0)}")
    else:
        print(f"   ❌ No se recibieron content_cards")
        print(f"   Respuesta: {data3.get('response_type')}")
    
    # Paso 4: Enviar consulta de texto
    print("\n4️⃣ Enviando consulta de texto...")
    payload4 = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "message": "Necesito materiales sobre matemáticas para 5to grado"
    }
    
    response4 = requests.post(f"{BASE_URL}/chat", json=payload4)
    data4 = response4.json()
    
    print(f"   Status: {response4.status_code}")
    print(f"   Tipo: {data4.get('response_type')}")
    
    if data4.get('response_type') == 'content_cards':
        print("   ✅ ¡Éxito! Se recibieron content_cards")
        content_data = data4.get('data', {})
        print(f"   Intro text: {content_data.get('intro_text', 'Sin intro')}")
        print(f"   Total results: {content_data.get('total_results', 0)}")
    else:
        print(f"   ❌ No se recibieron content_cards")
        print(f"   Respuesta: {data4.get('response_type')}")
    
    print("\n" + "=" * 60)
    print("🏁 Prueba de content_cards completada")

if __name__ == "__main__":
    test_content_cards() 