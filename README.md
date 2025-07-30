# RedMag Chatbot API

API de chatbot inteligente construida con FastAPI y Vertex AI Vector Search, diseñada para ejecutarse en Google Cloud Run.

## Características

- **Chatbot Inteligente**: Utiliza Vertex AI Vector Search para recuperar contexto relevante
- **Gestión de Documentos**: Operaciones CRUD completas en la base de datos vectorial
- **Integración con CMS**: Migración automática de contenido desde WordPress
- **🌐 Integración de Web Scraping con Turbo-Firecrawl**: Extracción automática de contenido web
- **Despliegue Automático**: CI/CD con GitHub Actions para Google Cloud Run
- **Escalabilidad**: Optimizado para Cloud Run con auto-scaling

## Estructura del Proyecto

```
redmag-chatbot/
├── modules/                    # Módulos de Vertex AI Vector Search
│   ├── __init__.py
│   ├── config.py              # Configuración y autenticación
│   ├── embeddings.py          # Generación de embeddings
│   ├── vector_search.py       # Operaciones de vector search
│   └── cms_integration.py     # Integración con CMS
├── turbo-firecrawl/           # Módulo de web scraping
│   └── ...                   # Componentes de Firecrawl
├── .github/workflows/
│   └── cicd.yaml             # Pipeline de CI/CD
├── app.py                    # API FastAPI principal
├── Dockerfile                # Configuración de Docker
├── requirements.txt          # Dependencias de Python
├── env_template.txt          # Plantilla de variables de entorno
└── README.md                # Este archivo
```

## Configuración

### 1. Variables de Entorno

Copia `env_template.txt` a `.env` y configura las variables:

```bash
cp env_template.txt .env
```

Configura las siguientes variables en `.env`:

```env
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT_ID=tu-proyecto-id
GOOGLE_CLOUD_LOCATION=us-central1

# Service Account Authentication
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json

# Vertex AI Vector Search
VECTOR_INDEX_ID=tu-index-id
VECTOR_ENDPOINT_ID=tu-endpoint-id

# Embedding Model
EMBEDDING_MODEL=textembedding-gecko@003
EMBEDDING_BATCH_SIZE=100
EMBEDDING_MAX_TOKENS=3072

# Processing Configuration
BATCH_SIZE=100
MAX_RETRIES=3
REQUEST_TIMEOUT=300

# Logging
LOG_LEVEL=INFO

# Web Scraping
FIRECRAWL_API_KEY=your-firecrawl-api-key

### 2. Service Account

1. Crea un service account en Google Cloud Console
2. Asigna los siguientes roles:
   - Vertex AI User
   - Cloud Run Admin
   - Storage Admin
3. Descarga la clave JSON y configúrala en `GOOGLE_APPLICATION_CREDENTIALS`

### 3. Vertex AI Vector Search

1. Crea un Vector Index en Vertex AI
2. Configura un Index Endpoint
3. Actualiza los IDs en las variables de entorno

### 4. Web Scraping con Firecrawl

1. Obtén una clave API de Firecrawl
2. Configura la variable `FIRECRAWL_API_KEY` en tu archivo `.env`
3. El módulo turbo-firecrawl se encargará automáticamente de la rotación de claves y reintentos

## Instalación Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la API localmente
python app.py
```

La API estará disponible en `http://localhost:8080`

## Endpoints de la API

### Chatbot

- `POST /chat` - Endpoint principal del chatbot
- `GET /health` - Verificación de estado del servicio

### Gestión de Documentos

- `POST /documents` - Insertar documento
- `PUT /documents/{document_id}` - Actualizar documento
- `DELETE /documents/{document_id}` - Eliminar documento

### Búsqueda

- `POST /search` - Buscar documentos similares

### CMS Integration

- `POST /cms/migrate` - Migrar contenido desde CMS

### Web Scraping

- `POST /scrape` - Extraer contenido de URLs web

## Ejemplos de Uso

### Chatbot

```bash
curl -X POST "http://localhost:8080/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "¿Qué servicios ofrecen?",
    "context_limit": 5,
    "user_id": "user123"
  }'
```

### Insertar Documento

```bash
curl -X POST "http://localhost:8080/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_001",
    "content": "Ofrecemos servicios de consultoría en tecnología...",
    "metadata": {
      "title": "Servicios de Consultoría",
      "category": "servicios"
    }
  }'
```

### Buscar Documentos

```bash
curl -X POST "http://localhost:8080/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "servicios de consultoría",
    "num_neighbors": 10
  }'
```

### Extraer Contenido Web

```bash
curl -X POST "http://localhost:8080/scrape" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://ejemplo.com/articulo",
    "extract_structured_data": true
  }'
```

## Despliegue en Google Cloud Run

### Configuración de GitHub Secrets

Configura los siguientes secrets en tu repositorio de GitHub:

- `GCP_PROJECT_ID`: ID de tu proyecto de Google Cloud
- `GCP_SA_KEY`: Clave JSON del service account (contenido completo del archivo)

### Despliegue Automático

El pipeline de CI/CD se ejecuta automáticamente en cada push a `main` o `master`:

1. **Test**: Verifica que el código se importe correctamente
2. **Build**: Construye la imagen Docker
3. **Deploy**: Despliega en Google Cloud Run

### Despliegue Manual

```bash
# Autenticarse con Google Cloud
gcloud auth login

# Configurar proyecto
gcloud config set project YOUR_PROJECT_ID

# Construir y desplegar
gcloud run deploy redmag-chatbot-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

## Configuración de Vertex AI

### Crear Vector Index

1. Ve a Vertex AI en Google Cloud Console
2. Navega a Vector Search
3. Crea un nuevo index con:
   - **Dimension**: 768 (para textembedding-gecko@003)
   - **Distance Measure**: COSINE_DISTANCE
   - **Algorithm**: TREE_AH

### Configurar Endpoint

1. Crea un Index Endpoint
2. Despliega el index en el endpoint
3. Configura las variables de entorno con los IDs correspondientes

## 🌐 Integración de Web Scraping con Turbo-Firecrawl

Este proyecto incorpora el módulo turbo-firecrawl para extender sus capacidades de ingesta de datos, permitiendo el web scraping y la extracción de información estructurada de URLs externas.

### Funcionalidad

El módulo turbo-firecrawl utiliza la API de Firecrawl para:
- Scrapear el contenido textual de páginas web enlazadas desde el CMS
- Extraer datos estructurados de URLs específicas (si es necesario para futuros casos de uso)
- Manejar automáticamente la rotación de claves API y reintentos para operaciones robustas

### Configuración

Para habilitar esta funcionalidad, asegúrate de que la variable de entorno `FIRECRAWL_API_KEY` esté configurada en tu archivo `.env` con una clave API válida de Firecrawl.

### Estructura

El módulo turbo-firecrawl se encuentra en su propio directorio (`./turbo-firecrawl/`), lo que facilita su reutilización y mantenimiento como un componente independiente.

## Monitoreo y Logs

La aplicación incluye logging configurado que se integra con Google Cloud Logging:

- **INFO**: Operaciones normales
- **WARNING**: Advertencias
- **ERROR**: Errores que requieren atención
- **DEBUG**: Información detallada

## Troubleshooting

### Problemas Comunes

1. **Error de autenticación**:
   - Verifica que las credenciales del service account estén configuradas
   - Asegúrate de que el proyecto tenga las APIs habilitadas

2. **Error de configuración**:
   - Verifica que todas las variables de entorno estén configuradas
   - Valida que los IDs del index y endpoint sean correctos

3. **Error de despliegue**:
   - Verifica que los GitHub secrets estén configurados correctamente
   - Revisa los logs de GitHub Actions para más detalles

### Logs de Debug

Para habilitar logs detallados localmente:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT.

## Soporte

Para soporte técnico o preguntas, crear un issue en el repositorio. 