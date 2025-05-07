import logging
from fastapi import FastAPI
# Ensure logging is configured when the app starts
from app.core.logging_config import get_logger # Assuming basic config runs on import
from app.core.config import settings # Import settings to check if they load

logger = get_logger(__name__)

app = FastAPI(title="Semantic Q/A API")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Semantic Q/A API...")
    logger.info(f"Qdrant Host: {settings.qdrant_host}") # Test settings loaded
    # Add any async startup logic here, e.g., initializing clients

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Semantic Q/A API...")
    # Add cleanup logic here

@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the Semantic Q/A API!"}

# Include routers later
# from app.api.v1.endpoints import documents, query
# app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
# app.include_router(query.router, prefix="/api/v1", tags=["Query"])