
import logging
from typing import List
from openai import OpenAI, APIError
from app.core.config import settings
from app.core.exceptions import EmbeddingError
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.provider = settings.embedding_provider.lower()
        self.model_name = settings.embedding_model_name

        if self.provider == "local_sentence_transformer":
            if SentenceTransformer is None:
                raise ImportError("SentenceTransformers library is required for local embeddings but not installed.")
            try:
                # The model will be downloaded from Hugging Face Hub automatically
                # the first time it's initialized and cached locally
                self.local_model = SentenceTransformer(self.model_name, device='cpu')
                logger.info(f"EmbeddingService initialized with local SentenceTransformer model: {self.model_name} on device 'cpu'")
            except Exception as e:
                logger.error(f"Failed to load local SentenceTransformer model '{self.model_name}': {e}")
                raise EmbeddingError(f"Failed to load local model: {e}")
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        try:
            if self.provider == "local_sentence_transformer":
                # The encode method returns a list of numpy arrays by default, convert to list of lists
                embeddings_np = self.local_model.encode(texts, convert_to_tensor=False) # convert_to_tensor=False gives numpy arrays
                embeddings = [emb.tolist() for emb in embeddings_np]
                logger.info(f"Successfully generated {len(embeddings)} embeddings using local model.")
            else:
                # Should be caught in __init__
                raise ValueError(f"Unsupported embedding provider in embed_texts: {self.provider}")
            return embeddings
        except Exception as e:
            logger.error(f"An unexpected error occurred during embedding with provider '{self.provider}': {e}")
            raise EmbeddingError(f"Failed to embed texts: {e}")