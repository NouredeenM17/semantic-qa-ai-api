
import logging
from fastapi import APIRouter, Depends, HTTPException
from app.api.schemas import QueryRequest, QueryResponse
from app.services.qa_service import QAService
from app.services.embedding_service import EmbeddingService # Needed by QAService.answer_query
from app.services.vector_store import VectorStoreService # Needed by QAService.answer_query
from app.api.deps import get_qa_service, get_embedding_service, get_vector_store_service
from app.core.exceptions import DocumentProcessingError # Catch potential errors from services

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/query", response_model=QueryResponse)
async def perform_query(
    request: QueryRequest, # Request body parsed into QueryRequest schema
    qa_s: QAService = Depends(get_qa_service),
    embedding_s: EmbeddingService = Depends(get_embedding_service), # Injected for answer_query
    vector_s: VectorStoreService = Depends(get_vector_store_service) # Injected for answer_query
):
    """
    Accepts a natural language query and returns a context-based answer
    along with source document information.
    """
    try:
        logger.info(f"Received query request: {request.query} with top_k={request.top_k_retrieval}, threshold={request.score_threshold}")
        
        # The QAService.answer_query method handles the full pipeline
        result = qa_s.answer_query(
            query=request.query,
            embedding_service=embedding_s,
            vector_store_service=vector_s,
            # collection_name=request.collection_name, # If you add this to QueryRequest
            top_k_retrieval=request.top_k_retrieval,
            score_threshold=request.score_threshold
        )
        
        # The result from answer_query is already a dict matching QueryResponse structure
        # but FastAPI will re-validate it against the response_model.
        # We can directly return it, or construct QueryResponse explicitly for clarity/safety.
        if "Error:" in result["answer"]: # Check for error messages from the service
             # Log the error specifically if it came from the service layer as a non-exception
            logger.error(f"Query processing returned an error in the answer field: {result['answer']}")
            # Depending on how service errors are structured, you might raise HTTPException here
            # For now, we assume service layer returns a dict that fits QueryResponse even on error

        return QueryResponse(**result) # Unpack the dict into the Pydantic model

    except DocumentProcessingError as e: # Example of catching a specific custom error
        logger.error(f"A document processing error occurred during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing your query.")