# MentorIA Chatbot API

API para un chatbot educativo personalizado con memoria a largo y corto plazo, integrado con Vertex AI y BigQuery, diseñado para el despliegue en Google Cloud Run.

## 🚀 Características

- **Arquitectura de Agentes Jerárquicos**: Sistema de dos agentes (Router Rápido y Analista Experto) para procesamiento inteligente de consultas
- **Búsqueda Vectorial**: Integración con Vertex AI Vector Search para búsqueda semántica de contenido educativo
- **Memoria Persistente**: Almacenamiento de perfiles de usuario y contexto de conversación en BigQuery
- **Gestión de Contenido**: Endpoints para crear, leer, actualizar y eliminar MEDs y Planeaciones
- **Autenticación**: Integración con APIs externas para autenticación y obtención de contenido
- **Escalabilidad**: Diseñado para despliegue en Google Cloud Run con auto-scaling

## 🏗️ Arquitectura

El proyecto sigue una arquitectura limpia con separación de responsabilidades:

```
src/
├── adapters/          # Capa de acceso a datos
│   ├── agents/        # Agentes de IA
│   ├── bigquery_adapter.py
│   ├── vector_search_adapter.py
│   └── gemini_adapter.py
├── controllers/       # Controladores HTTP
├── models/           # Modelos de datos
├── services/         # Lógica de negocio
└── config.py         # Configuración
```

### Agentes Jerárquicos

1. **RouterAgent**: Agente rápido que realiza triaje inicial de consultas
2. **ComplexQueryAgent**: Agente experto para análisis profundo y síntesis

## 📋 Prerrequisitos

- Python 3.11+
- Google Cloud Platform account
- Vertex AI Vector Search index configurado
- BigQuery dataset configurado
- Gemini API key

## 🔧 Configuración

### Variables de Entorno

Crear un archivo `.env` con las siguientes variables:

```env
# Google Cloud
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_CLOUD_LOCATION=us-east1
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# BigQuery
BIGQUERY_USERS_TABLE=project.dataset.users
BIGQUERY_MESSAGES_TABLE=project.dataset.messages
BIGQUERY_CONTEXT_TABLE=project.dataset.conversation_context

# Vertex AI Vector Search
VECTOR_INDEX_ID=your-index-id
VECTOR_ENDPOINT_ID=your-endpoint-id
DEPLOYED_INDEX_ID=your-deployed-index-id

# Gemini AI
GEMINI_API_KEY=your-gemini-api-key

# External APIs
API_USERNAME=your-api-username
API_PASSWORD=your-api-password
FIRECRAWL_API_KEYS=your-firecrawl-api-key

# Storage
GCS_BUCKET_NAME=your-bucket-name
STATIC_FILES_PATH=path/to/static/files
KNOWLEDGE_BASE_NEM_PATH=knowledge_base_nem.json
SEP_KNOWLEDGE_BASE_PATH=sep_knowledge_base.json

# Logging
LOG_LEVEL=INFO

# Chat Configuration
MAX_MESSAGES_PER_CONVERSATION=20
MAX_HISTORY_CONTEXT=8

## 🚀 Despliegue en Google Cloud Run

### 1. Preparar el Proyecto

```bash
# Clonar el repositorio
git clone <repository-url>
cd redmag-chatbot

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar Google Cloud

```bash
# Configurar proyecto
gcloud config set project YOUR_PROJECT_ID

# Habilitar APIs necesarias
gcloud services enable run.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable bigquery.googleapis.com
```

### 3. Desplegar en Cloud Run

```bash
# Construir y desplegar
gcloud run deploy mentoria-chatbot \
  --source . \
  --platform managed \
  --region us-east1 \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_CLOUD_PROJECT_ID=YOUR_PROJECT_ID"
```

### 4. Configurar CI/CD (Opcional)

Para despliegue automático desde GitHub:

1. Conectar el repositorio a Cloud Build
2. Configurar trigger para despliegue automático en push a main
3. Configurar variables de entorno en Cloud Run

## 📚 Endpoints de la API

### Chat

- `POST /chat` - Procesar interacción del chat

### Contenido

#### MEDs
- `GET /content/meds` - Obtener contenido MEDs
- `POST /content/meds` - Crear nuevo contenido MED
- `PUT /content/meds/{med_id}` - Actualizar contenido MED
- `DELETE /content/meds/{med_id}` - Eliminar contenido MED

#### Planeaciones
- `GET /content/planeaciones` - Obtener contenido Planeaciones
- `POST /content/planeaciones` - Crear nuevo contenido Planeación
- `PUT /content/planeaciones/{planeacion_id}` - Actualizar contenido Planeación
- `DELETE /content/planeaciones/{planeacion_id}` - Eliminar contenido Planeación

### Utilidades

- `GET /` - Endpoint raíz
- `GET /health` - Health check

## 🔄 Flujo de Trabajo

1. **Recepción de Consulta**: El usuario envía un mensaje o datos estructurados
2. **Verificación de Límite**: Se verifica si la conversación ha alcanzado el límite de 20 mensajes
3. **Nueva Conversación**: Si se alcanza el límite, se crea automáticamente una nueva conversación
4. **Triaje Inicial**: RouterAgent analiza la consulta y decide la ruta
5. **Procesamiento**: Dependiendo del tipo de consulta:
   - Respuesta directa
   - Solicitud de información adicional
   - Análisis profundo con ComplexQueryAgent
6. **Búsqueda Vectorial**: Si es necesario, se realiza búsqueda semántica
7. **Respuesta**: Se genera y devuelve la respuesta apropiada
8. **Persistencia**: Se actualiza el contexto y perfil del usuario

### 📊 Gestión de Conversaciones

- **Límite de Mensajes**: Cada conversación tiene un límite máximo de 20 mensajes
- **Renovación Automática**: Al alcanzar el límite, se crea automáticamente una nueva conversación
- **Contexto Preservado**: El perfil del usuario se mantiene entre conversaciones
- **Historial Limitado**: Se mantienen solo los últimos 8 mensajes en el contexto para optimizar rendimiento

### 🤖 Personalidad Jarvis

- **Mensaje de Bienvenida**: Al iniciar una conversación, Jarvis presenta todas las opciones disponibles
- **Tono Formal y Respetuoso**: Usa "señor/señora" y español formal
- **Proactivo**: Anticipa necesidades y ofrece soluciones
- **Técnico pero Accesible**: Usa términos técnicos cuando es apropiado pero explica claramente
- **Menú Interactivo**: Presenta botones con todas las funcionalidades disponibles
- **Consulta Personalizada**: Opción para escribir consultas específicas directamente

### 📋 Opciones del Menú Principal

- **📚 Ayuda con Planeaciones**: Crear y mejorar planeaciones didácticas
- **📖 Materiales Educativos (MEDs)**: Buscar y crear materiales educativos
- **🎯 Evaluación y Diagnóstico**: Herramientas de evaluación y diagnóstico educativo
- **🔧 Metodologías de Enseñanza**: Estrategias y metodologías pedagógicas
- **📋 Programas Analíticos**: Ayuda con programas analíticos y secuencias
- **❓ Preguntas Generales**: Consultas generales sobre educación
- **⚙️ Configurar Perfil**: Actualizar información del perfil docente
- **✍️ Escribir Consulta Personalizada**: Escribir consultas específicas directamente

### 🔄 Flujo de Consulta Personalizada

1. **Selección de "Otro"**: El usuario selecciona "✍️ Escribir Consulta Personalizada"
2. **Solicitud de Input**: Jarvis solicita al usuario escribir su consulta específica
3. **Procesamiento**: La consulta se procesa a través del sistema de agentes jerárquicos
4. **Respuesta**: Se genera una respuesta personalizada basada en la consulta del usuario

### 📝 Tipos de Respuesta

- **`welcome`**: Mensaje de bienvenida con opciones del menú
- **`buttons`**: Preguntas de configuración de perfil
- **`text_input`**: Solicitud para escribir consulta personalizada
- **`text`**: Respuesta de texto del agente
- **`vector_search`**: Resultados de búsqueda en la base de datos vectorial

## 🧪 Testing

```bash
# Ejecutar tests
python -m pytest tests/

# Ejecutar con coverage
python -m pytest --cov=src tests/
```

## 📊 Monitoreo

- **Logs**: Cloud Logging integrado
- **Métricas**: Cloud Monitoring para métricas de rendimiento
- **Trazabilidad**: Cada interacción se registra en BigQuery

## 🔒 Seguridad

- Autenticación mediante service accounts
- Variables de entorno para configuración sensible
- Validación de entrada con Pydantic
- Manejo seguro de errores

## 🤝 Contribución

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🆘 Soporte

Para soporte técnico, contactar al equipo de desarrollo o crear un issue en el repositorio. 