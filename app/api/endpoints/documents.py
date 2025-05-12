
import logging
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Form
from app.api.schemas import UploadResponse
from app.services.document_processor import process_and_index_pdf
from app.core.exceptions import DocumentProcessingError, PDFParsingError
from app.api.deps import get_embedding_service, get_vector_store_service
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService

router = APIRouter()
logger = logging.getLogger(__name__)

# Define a background task function that can be called
async def background_process_pdf(
    file_content: bytes,
    filename: str,
    author: Optional[str], # Added author
    embedding_service: EmbeddingService,
    vector_store_service: VectorStoreService
):
    """
    Helper function to be run in the background for processing a single PDF.
    """
    try:
        logger.info(f"Background task started for: {filename}")
        doc_id = await process_and_index_pdf(
            pdf_bytes=file_content,
            filename=filename,
            embedding_service=embedding_service,
            vector_store_service=vector_store_service,
            author=author # Pass author along
        )
        logger.info(f"Background task completed for: {filename}, Document ID: {doc_id}")
    except PDFParsingError as e:
        logger.error(f"PDF Parsing Error in background for {filename}: {e}")
        # How to report this back to user? For now, just log.
        # Could write to a DB, notification system, etc.
    except DocumentProcessingError as e:
        logger.error(f"Document Processing Error in background for {filename}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in background processing for {filename}: {e}", exc_info=True)


@router.post("/upload", response_model=UploadResponse, status_code=202) # 202 Accepted for background tasks
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="One or more PDF files to upload."),
    author: Optional[str] = Form(None, description="Optional author name for the documents."), # Author as a form field
    # Service dependencies injected by FastAPI
    embedding_s: EmbeddingService = Depends(get_embedding_service),
    vector_s: VectorStoreService = Depends(get_vector_store_service),
):
    """
    Uploads one or more PDF documents for processing and indexing.
    Processing happens in the background.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    processed_doc_ids: List[str] = []
    failed_files: List[str] = []
    
    for file in files:
        if not file.filename: # Should not happen with FastAPI's UploadFile but good check
            logger.warning("Received a file without a filename.")
            failed_files.append("Unnamed file") # Or handle as you see fit
            continue

        if file.content_type != "application/pdf":
            logger.warning(f"Invalid file type for {file.filename}: {file.content_type}")
            failed_files.append(file.filename)
            # Optionally raise HTTPException here if you want to fail the whole request
            # raise HTTPException(status_code=400, detail=f"Invalid file type for {file.filename}: {file.content_type}. Only PDFs are allowed.")
            continue # Skip non-PDF files

        try:
            file_content = await file.read()
            # Add the processing to background tasks
            background_tasks.add_task(
                background_process_pdf,
                file_content,
                file.filename,
                author, # Pass author to background task
                embedding_s,
                vector_s
            )
            # For now, we don't have the doc_id immediately as it's backgrounded.
            # We could return a task ID or a temporary reference if needed.
            # For simplicity, let's just acknowledge the file was queued.
            # If you need individual doc_ids back, you'd need a more complex system
            # (e.g., background task writes ID to a DB, client polls an endpoint).
            # Let's consider a conceptual "file_reference_id" or just the filename for now.
            # processed_doc_ids.append(file.filename) # Using filename as a temporary reference
            logger.info(f"File '{file.filename}' queued for background processing.")

        except Exception as e:
            logger.error(f"Failed to read or queue file {file.filename} for processing: {e}")
            failed_files.append(file.filename)
        finally:
            await file.close() # Important to close the file

    if not processed_doc_ids and not failed_files and files:
         # This case means all files were valid and queued. Let's make a list of queued files.
        processed_doc_ids = [f.filename for f in files if f.filename not in failed_files and f.content_type == "application/pdf"]


    if not processed_doc_ids and failed_files:
         return UploadResponse(
            message="Some files failed validation. No files queued for processing.",
            failed_files=failed_files
        )
    
    return UploadResponse(
        message=f"{len(processed_doc_ids)} file(s) accepted and queued for background processing.",
        document_ids=processed_doc_ids, # These are filenames of queued files
        failed_files=failed_files
    )