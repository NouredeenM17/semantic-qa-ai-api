# app/api/deps.py
from functools import lru_cache # For caching service instances
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.qa_service import QAService
from app.core.config import settings # If services need settings directly (they do)

# Using lru_cache to create singleton-like instances for services
# This means each service will be initialized only once per application lifecycle.

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    # EmbeddingService init might raise ValueError if API key is missing
    # This will be caught by FastAPI if it happens during dependency resolution
    return EmbeddingService()

@lru_cache()
def get_vector_store_service() -> VectorStoreService:
    # VectorStoreService init might raise VectorStoreError if Qdrant connection fails
    return VectorStoreService() # It creates its own QdrantClient

@lru_cache()
def get_qa_service() -> QAService:
    # QAService init might raise ValueError or NotImplementedError
    return QAService()

# Example usage in an endpoint:
# from fastapi import APIRouter, Depends
# from app.api.deps import get_qa_service
# from app.services.qa_service import QAService
#
# router = APIRouter()
#
# @router.post("/some_path")
# async def some_endpoint(qa_s: QAService = Depends(get_qa_service)):
#     # use qa_s here
#     pass