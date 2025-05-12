
import logging
from contextlib import asynccontextmanager # For lifespan events in newer FastAPI/Starlette

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware # If you need CORS

# Import your API routers
from app.api.endpoints import documents as documents
from app.api.endpoints import query as query

# Import settings and service getters for lifespan
from app.core.config import settings
from app.core.logging_config import LOGGING_LEVEL, LOGGING_FORMAT # Or your get_logger setup
from app.api.deps import get_vector_store_service # To initialize collection on startup


# Setup logging
# This basic config should be fine for now.
# For more advanced, you might use logging.config.dictConfig
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)


# Lifespan manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("Application startup...")
    logger.info(f"Running on Qdrant host: {settings.qdrant_host}")
    
    # Initialize Qdrant collection on startup
    try:
        vector_store_service = get_vector_store_service() # Get instance
        await vector_store_service.initialize_collection_if_not_exists(
            collection_name=settings.qdrant_collection_name,
            vector_size=settings.embedding_dim
        )
        logger.info(f"Qdrant collection '{settings.qdrant_collection_name}' initialized/verified.")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collection during startup: {e}", exc_info=True)
        # Depending on severity, you might want to prevent app startup or handle differently
        # For now, we log the error and continue.

    # You could initialize other ML models or clients here if needed
    # e.g., embedding_service = get_embedding_service() to preload models (if applicable)
    # qa_service = get_qa_service()

    yield # Application runs after this yield

    # --- Shutdown ---
    logger.info("Application shutdown...")
    # Add any cleanup tasks here, e.g., closing database connections if not managed by client
    # qdrant_client used by VectorStoreService might have a close() method,
    # but typically it's managed by the client library's lifecycle.
    # If get_vector_store_service().client has a close method:
    # try:
    #     get_vector_store_service().client.close()
    #     logger.info("Qdrant client closed.")
    # except Exception as e:
    #     logger.error(f"Error closing Qdrant client: {e}")


# Create FastAPI app instance with lifespan manager
app = FastAPI(
    title="Semantic Q/A API",
    description="API for semantic question answering over uploaded documents.",
    version="0.1.0",
    lifespan=lifespan # Assign the lifespan context manager
)

# --- Middleware ---
# Example: CORS (Cross-Origin Resource Sharing)
# Adjust origins as needed for your frontend if it's on a different domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Include API Routers ---
# Create a main API router for versioning (optional but good practice)
api_router = APIRouter(prefix="/api")
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_router.include_router(query.router, prefix="/query", tags=["Query"])

app.include_router(api_router) # Include the versioned router into the main app


# --- Root endpoint (optional) ---
@app.get("/", tags=["Root"])
async def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the Semantic Q/A API! Visit /docs for API documentation."}
