"""
MentorIA Chatbot API - Main Application Entry Point

FastAPI application for the MentorIA educational chatbot with hierarchical agents,
vector search, and BigQuery integration for Google Cloud Run deployment.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.controllers.chat_controller import ChatController
from src.controllers.content_controller import ContentController
from src.models.request_models import ChatRequest, UserDataInput
from src.models.response_models import ChatResponse
from src.services.dependency_injection import get_chat_controller, get_content_controller
from src.config import config

logging.basicConfig(level=getattr(logging, config.log_level, "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MentorIA Chatbot API",
    description="Educational chatbot with hierarchical agents, vector search, and BigQuery integration",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", summary="Root Endpoint")
async def root():
    """Root endpoint with welcome message."""
    return {"message": "MentorIA Chatbot API v2.0.0"}

@app.get("/health", summary="Health Check")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/chat", response_model=ChatResponse, summary="Process Chat Interaction")
async def chat(
    request: ChatRequest,
    chat_controller: ChatController = Depends(get_chat_controller)
):
    """
    Main chat endpoint that handles user interactions.
    
    Processes text messages and structured data inputs through the hierarchical agent system.
    """
    if not request.message and not request.user_data:
        raise HTTPException(status_code=400, detail="Must provide 'message' or 'user_data'")

    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        response_data = await chat_controller.handle_interaction(
            user_id=request.user_id,
            conversation_id=conversation_id,
            message=request.message,
            user_data_input=request.user_data
        )

        # Usar el conversation_id actualizado si se creó una nueva conversación
        final_conversation_id = response_data.get("conversation_id", conversation_id)
        
        return ChatResponse(
            conversation_id=final_conversation_id,
            response_type=response_data["type"],
            data=response_data["data"]
        )

    except Exception as e:
        logger.error(f"Critical error in chat endpoint for user {request.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error processing request")

@app.get("/content/meds", summary="Get MEDs Content")
async def get_meds_content(
    content_controller: ContentController = Depends(get_content_controller)
):
    """Retrieve MEDs content from the vector database."""
    try:
        return await content_controller.get_meds_content()
    except Exception as e:
        logger.error(f"Error retrieving MEDs content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving content")

@app.post("/content/meds", summary="Create MED Content")
async def create_med_content(
    content_data: Dict[str, Any],
    content_controller: ContentController = Depends(get_content_controller)
):
    """Create new MED content in the vector database."""
    try:
        return await content_controller.create_med_content(content_data)
    except Exception as e:
        logger.error(f"Error creating MED content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error creating content")

@app.put("/content/meds/{med_id}", summary="Update MED Content")
async def update_med_content(
    med_id: str,
    content_data: Dict[str, Any],
    content_controller: ContentController = Depends(get_content_controller)
):
    """Update existing MED content in the vector database."""
    try:
        return await content_controller.update_med_content(med_id, content_data)
    except Exception as e:
        logger.error(f"Error updating MED content {med_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error updating content")

@app.delete("/content/meds/{med_id}", summary="Delete MED Content")
async def delete_med_content(
    med_id: str,
    content_controller: ContentController = Depends(get_content_controller)
):
    """Delete MED content from the vector database."""
    try:
        return await content_controller.delete_med_content(med_id)
    except Exception as e:
        logger.error(f"Error deleting MED content {med_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error deleting content")

@app.get("/content/planeaciones", summary="Get Planeaciones Content")
async def get_planeaciones_content(
    content_controller: ContentController = Depends(get_content_controller)
):
    """Retrieve Planeaciones content from the vector database."""
    try:
        return await content_controller.get_planeaciones_content()
    except Exception as e:
        logger.error(f"Error retrieving Planeaciones content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving content")

@app.post("/content/planeaciones", summary="Create Planeacion Content")
async def create_planeacion_content(
    content_data: Dict[str, Any],
    content_controller: ContentController = Depends(get_content_controller)
):
    """Create new Planeacion content in the vector database."""
    try:
        return await content_controller.create_planeacion_content(content_data)
    except Exception as e:
        logger.error(f"Error creating Planeacion content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error creating content")

@app.put("/content/planeaciones/{planeacion_id}", summary="Update Planeacion Content")
async def update_planeacion_content(
    planeacion_id: str,
    content_data: Dict[str, Any],
    content_controller: ContentController = Depends(get_content_controller)
):
    """Update existing Planeacion content in the vector database."""
    try:
        return await content_controller.update_planeacion_content(planeacion_id, content_data)
    except Exception as e:
        logger.error(f"Error updating Planeacion content {planeacion_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error updating content")

@app.delete("/content/planeaciones/{planeacion_id}", summary="Delete Planeacion Content")
async def delete_planeacion_content(
    planeacion_id: str,
    content_controller: ContentController = Depends(get_content_controller)
):
    """Delete Planeacion content from the vector database."""
    try:
        return await content_controller.delete_planeacion_content(planeacion_id)
    except Exception as e:
        logger.error(f"Error deleting Planeacion content {planeacion_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error deleting content")
