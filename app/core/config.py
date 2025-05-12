
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Load .env file located in the project root
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Qdrant Configuration
    qdrant_host: str = Field("localhost", alias='QDRANT_HOST')
    qdrant_port: int = Field(6333, alias='QDRANT_PORT')
    qdrant_collection_name: str = Field("semantic_qa_collection", alias='QDRANT_COLLECTION_NAME')
    qdrant_prefer_grpc: bool = Field(True, alias='QDRANT_PREFER_GRPC')

    # Embedding Model Configuration
    embedding_provider : str = Field("local_sentence_transformer", alias="EMBEDDING_PROVIDER")
    embedding_model_name: str = Field("sentence-transformers/all-MiniLM-L6-v2", alias='EMBEDDING_MODEL_NAME')
    embedding_dim: int = Field(384, alias='EMBEDDING_DIM')
    openai_api_key: str = Field(..., alias='OPENAI_API_KEY') # Make required
    # gemini_api_key: Optional[str] = Field(None, alias='GEMINI_API_KEY') # Add later if needed

    # LLM Configuration
    llm_provider: str = Field("openai", alias='LLM_PROVIDER')
    llm_model_name: str = Field("gpt-4-turbo-preview", alias='LLM_MODEL_NAME')
    llm_temperature: float = Field(0.1, alias='LLM_TEMPERATURE', ge=0.0, le=2.0)
    llm_max_tokens: int = Field(500, alias='LLM_MAX_TOKENS', gt=0)

    # Document Processing Configuration
    chunk_size: int = Field(700, alias='CHUNK_SIZE')
    chunk_overlap: int = Field(100, alias='CHUNK_OVERLAP')

# Create a single instance for the application to import
settings = Settings()

# Example usage:
# from app.core.config import settings
# print(settings.qdrant_host)
# print(settings.openai_api_key)