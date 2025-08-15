"""
Script para probar la funcionalidad de límite de conversaciones (20 mensajes máximo).
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_conversation_limit():
    """Probar el límite de 20 mensajes por conversación."""
    print("🧪 Probando límite de conversaciones (20 mensajes máximo)")
    print("=" * 60)
    
    user_id = "test_user_limit_123"
    conversation_id = None
    
    # Enviar 25 mensajes para probar el límite
    for i in range(25):
        print(f"\n📝 Mensaje {i+1}/25")
        
        payload = {
            "user_id": user_id,
            "message": f"Este es el mensaje número {i+1} de la conversación"
        }
        
        if conversation_id:
            payload["conversation_id"] = conversation_id
        
        try:
            response = requests.post(f"{BASE_URL}/chat", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                new_conversation_id = data.get("conversation_id")
                
                # Verificar si se creó una nueva conversación
                if conversation_id and new_conversation_id != conversation_id:
                    print(f"🔄 ¡Nueva conversación creada en mensaje {i+1}!")
                    print(f"   Conversación anterior: {conversation_id}")
                    print(f"   Nueva conversación: {new_conversation_id}")
                
                conversation_id = new_conversation_id
                print(f"   ✅ Respuesta recibida - Conversación: {conversation_id}")
                
                # Mostrar el tipo de respuesta
                response_type = data.get("response_type", "unknown")
                print(f"   📋 Tipo de respuesta: {response_type}")
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"   {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error de conexión: {e}")
        
        # Pequeña pausa entre mensajes
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("🏁 Prueba de límite de conversaciones completada")
    print(f"📊 Conversación final: {conversation_id}")

def test_conversation_reset():
    """Probar que se puede iniciar una nueva conversación manualmente."""
    print("\n🔄 Probando reinicio manual de conversación")
    print("=" * 60)
    
    user_id = "test_user_reset_456"
    
    # Primera conversación
    payload1 = {
        "user_id": user_id,
        "message": "Primera conversación"
    }
    
    try:
        response1 = requests.post(f"{BASE_URL}/chat", json=payload1)
        if response1.status_code == 200:
            data1 = response1.json()
            conv1_id = data1.get("conversation_id")
            print(f"✅ Primera conversación: {conv1_id}")
            
            # Segunda conversación (sin conversation_id para forzar nueva)
            payload2 = {
                "user_id": user_id,
                "message": "Segunda conversación (nueva)"
            }
            
            response2 = requests.post(f"{BASE_URL}/chat", json=payload2)
            if response2.status_code == 200:
                data2 = response2.json()
                conv2_id = data2.get("conversation_id")
                print(f"✅ Segunda conversación: {conv2_id}")
                
                if conv1_id != conv2_id:
                    print("🔄 ✅ Se creó una nueva conversación correctamente")
                else:
                    print("⚠️ Las conversaciones tienen el mismo ID")
                    
        else:
            print(f"❌ Error en primera conversación: {response1.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Función principal."""
    print("🚀 Iniciando pruebas de límite de conversaciones")
    
    # Esperar que el servidor esté listo
    print("⏳ Esperando que el servidor esté listo...")
    time.sleep(2)
    
    # Probar límite de conversaciones
    test_conversation_limit()
    
    # Probar reinicio manual
    test_conversation_reset()
    
    print("\n✅ Todas las pruebas completadas")

if __name__ == "__main__":
    main() 