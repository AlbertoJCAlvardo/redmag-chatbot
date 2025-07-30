"""
Chatbot API with Vertex AI Vector Search.

FastAPI application that provides chatbot functionality using
Vertex AI Vector Search for context retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from modules.vector_search import VectorSearchManager
from modules.embeddings import EmbeddingGenerator
from modules.cms_integration import CMSIntegration, WordPressConnector
from modules.config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RedMag Chatbot API",
    description="Chatbot API powered by Vertex AI Vector Search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
vector_manager = None
embedding_generator = None
cms_integration = None


# Pydantic models for request/response
class ChatRequest(BaseModel):
    """Request model for chat operations."""
    message: str = Field(..., description="User message")
    context_limit: int = Field(5, description="Number of context documents to retrieve")
    user_id: Optional[str] = Field(None, description="User identifier")


class ChatResponse(BaseModel):
    """Response model for chat operations."""
    success: bool
    message: str
    context_documents: List[Dict[str, Any]]
    total_context: int
    timestamp: str


class DocumentRequest(BaseModel):
    """Request model for document operations."""
    document_id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="Text content to be embedded and stored")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    success: bool
    message: str
    document_id: Optional[str] = None


class SearchRequest(BaseModel):
    """Request model for search operations."""
    query: str = Field(..., description="Search query text")
    num_neighbors: int = Field(10, description="Number of similar documents to return")
    filter_expression: Optional[str] = Field(None, description="Filter expression for search")


class SearchResponse(BaseModel):
    """Response model for search operations."""
    success: bool
    query: str
    results: List[Dict[str, Any]]
    total_results: int


class CMSMigrationRequest(BaseModel):
    """Request model for CMS migration."""
    cms_type: str = Field(..., description="Type of CMS (e.g., 'wordpress')")
    site_url: str = Field(..., description="CMS site URL")
    username: str = Field(..., description="CMS username")
    password: str = Field(..., description="CMS password")
    limit: Optional[int] = Field(None, description="Maximum number of content items to migrate")


class CMSMigrationResponse(BaseModel):
    """Response model for CMS migration."""
    success: bool
    message: str
    statistics: Optional[Dict[str, Any]] = None


# Dependency injection
def get_vector_manager() -> VectorSearchManager:
    """Get or create VectorSearchManager instance."""
    global vector_manager
    if vector_manager is None:
        vector_manager = VectorSearchManager()
    return vector_manager


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create EmbeddingGenerator instance."""
    global embedding_generator
    if embedding_generator is None:
        embedding_generator = EmbeddingGenerator()
    return embedding_generator


def get_cms_integration() -> CMSIntegration:
    """Get or create CMSIntegration instance."""
    global cms_integration
    if cms_integration is None:
        vector_mgr = get_vector_manager()
        cms_integration = CMSIntegration(vector_mgr)
    return cms_integration


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow().isoformat(),
        "service": "RedMag Chatbot API"
    }


# Chatbot endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    vector_mgr: VectorSearchManager = Depends(get_vector_manager)
):
    """
    Main chatbot endpoint that retrieves relevant context for user messages.
    
    Args:
        request: Chat request data
        vector_mgr: Vector search manager instance
        
    Returns:
        ChatResponse: Chat response with context documents
    """
    try:
        # Search for relevant documents
        results = vector_mgr.search_similar(
            query=request.message,
            num_neighbors=request.context_limit
        )
        
        # Format results for chatbot context
        context_documents = []
        for result in results:
            context_doc = {
                "document_id": result["document_id"],
                "relevance_score": 1.0 - result["distance"],  # Convert distance to similarity
                "metadata": result.get("metadata", {})
            }
            context_documents.append(context_doc)
        
        return ChatResponse(
            success=True,
            message=f"Retrieved {len(context_documents)} relevant documents",
            context_documents=context_documents,
            total_context=len(context_documents),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Document management endpoints
@app.post("/documents", response_model=DocumentResponse)
async def insert_document(
    request: DocumentRequest,
    vector_mgr: VectorSearchManager = Depends(get_vector_manager)
):
    """
    Insert a single document into the vector index.
    
    Args:
        request: Document request data
        vector_mgr: Vector search manager instance
        
    Returns:
        DocumentResponse: Operation result
    """
    try:
        success = vector_mgr.insert_document(
            document_id=request.document_id,
            content=request.content,
            metadata=request.metadata
        )
        
        return DocumentResponse(
            success=success,
            message="Document inserted successfully" if success else "Failed to insert document",
            document_id=request.document_id
        )
        
    except Exception as e:
        logger.error(f"Error inserting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/documents/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str,
    request: DocumentRequest,
    vector_mgr: VectorSearchManager = Depends(get_vector_manager)
):
    """
    Update an existing document in the vector index.
    
    Args:
        document_id: ID of the document to update
        request: Document request data
        vector_mgr: Vector search manager instance
        
    Returns:
        DocumentResponse: Operation result
    """
    try:
        success = vector_mgr.update_document(
            document_id=document_id,
            content=request.content,
            metadata=request.metadata
        )
        
        return DocumentResponse(
            success=success,
            message="Document updated successfully" if success else "Failed to update document",
            document_id=document_id
        )
        
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}", response_model=DocumentResponse)
async def delete_document(
    document_id: str,
    vector_mgr: VectorSearchManager = Depends(get_vector_manager)
):
    """
    Delete a document from the vector index.
    
    Args:
        document_id: ID of the document to delete
        vector_mgr: Vector search manager instance
        
    Returns:
        DocumentResponse: Operation result
    """
    try:
        success = vector_mgr.delete_document(document_id)
        
        return DocumentResponse(
            success=success,
            message="Document deleted successfully" if success else "Failed to delete document",
            document_id=document_id
        )
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Search endpoints
@app.post("/search", response_model=SearchResponse)
async def search_similar(
    request: SearchRequest,
    vector_mgr: VectorSearchManager = Depends(get_vector_manager)
):
    """
    Search for similar documents using a query string.
    
    Args:
        request: Search request data
        vector_mgr: Vector search manager instance
        
    Returns:
        SearchResponse: Search results
    """
    try:
        results = vector_mgr.search_similar(
            query=request.query,
            num_neighbors=request.num_neighbors,
            filter_expression=request.filter_expression
        )
        
        return SearchResponse(
            success=True,
            query=request.query,
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# CMS integration endpoints
@app.post("/cms/migrate", response_model=CMSMigrationResponse)
async def migrate_cms_content(
    request: CMSMigrationRequest,
    background_tasks: BackgroundTasks,
    cms_int: CMSIntegration = Depends(get_cms_integration)
):
    """
    Migrate content from CMS to vector database.
    
    Args:
        request: CMS migration request data
        background_tasks: FastAPI background tasks
        cms_int: CMS integration instance
        
    Returns:
        CMSMigrationResponse: Migration result
    """
    try:
        # Create appropriate CMS connector
        if request.cms_type.lower() == "wordpress":
            cms_connector = WordPressConnector(
                site_url=request.site_url,
                username=request.username,
                password=request.password
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported CMS type: {request.cms_type}")
        
        # Run migration in background
        def run_migration():
            try:
                result = cms_int.migrate_content(cms_connector, limit=request.limit)
                logger.info(f"Background migration completed: {result}")
            except Exception as e:
                logger.error(f"Background migration failed: {e}")
        
        background_tasks.add_task(run_migration)
        
        return CMSMigrationResponse(
            success=True,
            message="CMS migration started in background",
            statistics={"status": "started"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting CMS migration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting RedMag Chatbot API...")
    
    try:
        # Validate configuration
        config.validate()
        logger.info("Configuration validated successfully")
        
        # Initialize global instances
        get_vector_manager()
        get_embedding_generator()
        get_cms_integration()
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"API startup failed: {e}")
        raise


# Main entry point for running the API
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    ) 