
from pydantic import BaseModel, Field, conint
from typing import List, Optional, Any

# --- Schemas for Document Upload ---
class UploadResponse(BaseModel):
    message: str
    document_ids: List[str] = [] # IDs of successfully processed/queued documents
    failed_files: List[str] = [] # Names of files that failed initial validation/upload

# --- Schemas for Querying ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The natural language query to answer.")
    top_k_retrieval: Optional[conint(gt=0)] = Field(5, description="Number of relevant chunks to retrieve for context.")
    # conint(gt=0) ensures top_k is a positive integer
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score for retrieved chunks (0.0 to 1.0).")
    # You could add other parameters here like 'collection_name' if users can target specific collections.
    # For user-specific filtering (Phase X - Auth):
    # user_id: Optional[str] = None # This would be set by the backend after authentication

class SourceDocument(BaseModel):
    id: Optional[str] = Field(None, description="Unique ID of the source chunk.") # Chunk ID
    document_id: Optional[str] = Field(None, description="ID of the parent document.")
    title: str = Field(..., description="Title of the source document (e.g., filename).")
    page_number: Optional[int] = Field(None, description="Page number in the source document where the context was found.")
    score: Optional[float] = Field(None, description="Similarity score of the retrieved chunk.")
    text_preview: Optional[str] = Field(None, description="A short preview of the source text chunk.")
    # author: Optional[str] = None # If you decide to return author


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The LLM-generated answer to the query.")
    sources: List[SourceDocument] = Field([], description="List of source documents used to generate the answer.")
