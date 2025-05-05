
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Load .env file located in the project root
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Qdrant Configuration
    qdrant_host: str = Field("localhost", alias='QDRANT_HOST')
    qdrant_port: int = Field(6333, alias='QDRANT_PORT')
    qdrant_collection_name: str = Field("semantic_qa_collection", alias='QDRANT_COLLECTION_NAME')

    # Embedding Model Configuration
    embedding_model_name: str = Field("text-embedding-3-small", alias='EMBEDDING_MODEL_NAME')
    embedding_dimensions: int = Field(1536, alias='EMBEDDING_DIMENSIONS')
    openai_api_key: str = Field(..., alias='OPENAI_API_KEY') # Make required
    # gemini_api_key: Optional[str] = Field(None, alias='GEMINI_API_KEY') # Add later if needed

    # LLM Configuration
    llm_model_name: str = Field("gpt-4-turbo-preview", alias='LLM_MODEL_NAME')
    
    # Document Processing Configuration
    chunk_size: int = Field(700, alias='CHUNK_SIZE')
    chunk_overlap: int = Field(100, alias='CHUNK_OVERLAP')

# Create a single instance for the application to import
settings = Settings()

# Example usage:
# from app.core.config import settings
# print(settings.qdrant_host)
# print(settings.openai_api_key)