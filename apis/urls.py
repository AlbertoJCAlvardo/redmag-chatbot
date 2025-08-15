from fastapi import APIRouter
from .sockets.urls import router as socket_router

apis = APIRouter()

apis.include_router(socket_router, prefix="/ws")
