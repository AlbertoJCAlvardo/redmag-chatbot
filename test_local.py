"""
Script de prueba local para MentorIA Chatbot API.

Este script prueba la funcionalidad básica sin depender de BigQuery.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Probar el endpoint de health check."""
    print("🔍 Probando health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Health check: {response.status_code}")
        print(f"   Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_chat():
    """Probar el endpoint de chat."""
    print("\n💬 Probando chat endpoint...")
    try:
        payload = {
            "user_id": "test_user_123",
            "message": "Hola, necesito ayuda con planeaciones didácticas"
        }
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        print(f"✅ Chat endpoint: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return True
    except Exception as e:
        print(f"❌ Chat endpoint failed: {e}")
        return False

def test_content_endpoints():
    """Probar los endpoints de contenido."""
    print("\n📚 Probando endpoints de contenido...")
    
    # Probar obtener MEDs
    try:
        response = requests.get(f"{BASE_URL}/content/meds")
        print(f"✅ GET /content/meds: {response.status_code}")
    except Exception as e:
        print(f"❌ GET /content/meds failed: {e}")
    
    # Probar obtener Planeaciones
    try:
        response = requests.get(f"{BASE_URL}/content/planeaciones")
        print(f"✅ GET /content/planeaciones: {response.status_code}")
    except Exception as e:
        print(f"❌ GET /content/planeaciones failed: {e}")
    
    # Probar crear MED
    try:
        payload = {
            "title": "MED de Matemáticas 5to Grado",
            "description": "Material de apoyo para matemáticas",
            "content": "Contenido detallado del MED...",
            "tags": ["matemáticas", "5to", "primaria"]
        }
        response = requests.post(f"{BASE_URL}/content/meds", json=payload)
        print(f"✅ POST /content/meds: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"❌ POST /content/meds failed: {e}")
    
    # Probar crear Planeación
    try:
        payload = {
            "title": "Planeación de Ciencias Naturales",
            "description": "Planeación para ciencias naturales",
            "content": "Contenido de la planeación...",
            "tags": ["ciencias", "naturales", "primaria"]
        }
        response = requests.post(f"{BASE_URL}/content/planeaciones", json=payload)
        print(f"✅ POST /content/planeaciones: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"❌ POST /content/planeaciones failed: {e}")

def main():
    """Función principal de pruebas."""
    print("🚀 Iniciando pruebas de MentorIA Chatbot API")
    print("=" * 50)
    
    # Esperar un poco para que el servidor esté listo
    print("⏳ Esperando que el servidor esté listo...")
    time.sleep(2)
    
    # Ejecutar pruebas
    health_ok = test_health()
    if not health_ok:
        print("❌ El servidor no está respondiendo. Asegúrate de que esté ejecutándose.")
        return
    
    chat_ok = test_chat()
    test_content_endpoints()
    
    print("\n" + "=" * 50)
    print("🏁 Pruebas completadas")
    
    if health_ok and chat_ok:
        print("✅ Todas las pruebas básicas pasaron")
    else:
        print("⚠️ Algunas pruebas fallaron")

if __name__ == "__main__":
    main() 