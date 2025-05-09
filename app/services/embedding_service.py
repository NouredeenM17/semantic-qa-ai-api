
import logging
from typing import List
from openai import OpenAI, APIError # Import APIError for specific OpenAI error handling
from app.core.config import settings
from app.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment variables.")
        try:
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.model_name = settings.embedding_model_name
            logger.info(f"EmbeddingService initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            # This is a critical error, a placeholder if client init fails
            raise EmbeddingError(f"Failed to initialize OpenAI client: {e}")


    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a list of texts using OpenAI.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of vector embeddings (each embedding is a list of floats).

        Raises:
            EmbeddingError: If the embedding process fails.
        """
        if not texts:
            return []
        
        try:
            # The OpenAI Python library v1.0.0+ uses a different API structure
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
                # dimensions=settings.embedding_dim # Only for 'text-embedding-3-small' and 'text-embedding-3-large' if you want non-default dimensions
            )
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Successfully generated {len(embeddings)} embeddings.")
            return embeddings
        except APIError as e: # More specific OpenAI error
            logger.error(f"OpenAI API error during embedding: {e} - {e.message}")
            raise EmbeddingError(f"OpenAI API error: {e.message}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during embedding: {e}")
            raise EmbeddingError(f"Failed to embed texts: {e}")

# Optional: Create a single instance for easier dependency injection later if preferred,
# or instantiate it where needed. For services, explicit instantiation can be clearer.
# embedding_service_instance = EmbeddingService()