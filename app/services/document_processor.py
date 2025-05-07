
import fitz  # PyMuPDF
import logging
from typing import List, Tuple, Dict, Optional
from app.core.exceptions import PDFParsingError
from app.core.config import settings # For chunk_size, chunk_overlap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Extracts text from each page of a PDF.

    Args:
        pdf_bytes: The PDF file content as bytes.

    Returns:
        A list of tuples, where each tuple contains (page_number, page_text).
        Page numbers are 1-indexed.

    Raises:
        PDFParsingError: If the PDF is corrupt, password-protected, or text extraction fails.
    """
    pages_text: List[Tuple[int, str]] = []
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")

        if document.is_encrypted:
            # Attempt to authenticate with an empty password
            # PyMuPDF's authenticate() returns the number of permissions granted, or 0 if failed.
            if not document.authenticate(""): 
                logger.error("PDF is password-protected and authentication failed.")
                raise PDFParsingError("PDF is password-protected and requires a password to open.")

        for page_num_zero_indexed in range(len(document)):
            page = document.load_page(page_num_zero_indexed)
            text = page.get_text("text")
            if text:
                pages_text.append((page_num_zero_indexed + 1, text.strip()))
        
        if not pages_text and len(document) > 0:
            logger.warning("PDF parsed but no text could be extracted. It might be an image-only PDF or have non-standard text encoding.")

        logger.info(f"Successfully extracted text from {len(pages_text)} pages.")
        return pages_text

    except RuntimeError as e: # Catch PyMuPDF's common operational errors
        logger.error(f"PyMuPDF runtime error during PDF parsing: {e}")
        # Check if the error message indicates a password issue, though the explicit check above should catch it.
        if "password" in str(e).lower():
             raise PDFParsingError(f"Failed to parse PDF, potentially due to encryption or corruption: {e}")
        raise PDFParsingError(f"Failed to parse PDF (PyMuPDF runtime error): {e}")
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred during PDF parsing: {e}", exc_info=True)
        raise PDFParsingError(f"An unexpected error occurred during PDF parsing: {e}")


def chunk_text(
    pages_data: List[Tuple[int, str]],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> List[Dict]:
    """
    Chunks text from extracted PDF pages.

    Args:
        pages_data: A list of (page_number, page_text) tuples.
        chunk_size: The target size for each chunk (in characters or tokens,
                    depending on splitter's behavior).
        chunk_overlap: The overlap between consecutive chunks.

    Returns:
        A list of dictionaries, where each dictionary represents a chunk
        and contains 'text', 'page_number', and 'chunk_index_in_page'.
    """
    all_chunks: List[Dict] = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # Measures chunk size by number of characters
        is_separator_regex=False, # Treats separators literally
    )

    doc_chunk_index = 0 # Overall chunk index for the document
    for page_number, page_text in pages_data:
        if not page_text.strip(): # Skip empty pages
            continue

        # Langchain's splitter works on "Document" objects or simple strings.
        # We can create temporary Langchain Document objects to preserve metadata easily.
        # Or, split text directly and manage metadata manually.
        # For simplicity with RecursiveCharacterTextSplitter, using its Document structure:
        
        langchain_docs = [LangchainDocument(page_content=page_text, metadata={"page_number": page_number})]
        split_texts = text_splitter.split_documents(langchain_docs)

        for i, doc_chunk in enumerate(split_texts):
            all_chunks.append({
                "text": doc_chunk.page_content,
                "page_number": doc_chunk.metadata["page_number"],
                "chunk_index_in_doc": doc_chunk_index, # Unique index for this chunk within the entire document
                # "chunk_index_in_page": i # Optional: if you need index within the page
            })
            doc_chunk_index += 1
    
    logger.info(f"Chunked text into {len(all_chunks)} chunks.")
    return all_chunks


    async def process_and_index_pdf(
        pdf_bytes: bytes,
        filename: str,
        embedding_service: EmbeddingService, # Pass as dependency
        vector_store_service: VectorStoreService, # Pass as dependency
        author: Optional[str] = None,
        collection_name: Optional[str] = None, # Allow overriding default collection
    ) -> str:
        """
        Orchestrates the PDF processing and indexing pipeline.
        1. Extracts text from PDF.
        2. Chunks the extracted text.
        3. Generates embeddings for chunks.
        4. Upserts chunks and embeddings to the vector store.

        Args:
            pdf_bytes: The PDF file content as bytes.
            filename: The original name of the PDF file (used as title).
            embedding_service: An instance of EmbeddingService.
            vector_store_service: An instance of VectorStoreService.
            author: Optional author of the document.
            collection_name: Optional name of the Qdrant collection to use.

        Returns:
            The unique ID generated for the processed document.

        Raises:
            DocumentProcessingError: If any step in the pipeline fails.
        """
        doc_id = str(uuid.uuid4())
        effective_collection_name = collection_name or settings.qdrant_collection_name
        logger.info(f"Starting processing for document: {filename}, assigned ID: {doc_id}")

        try:
            # 1. Ensure Qdrant collection exists (idempotent)
            # This should be called once at application startup or ensured by deployment.
            # For robustness, we can call it here too, but be mindful of performance if called frequently.
            # Making it async because initialize_collection_if_not_exists is async
            await vector_store_service.initialize_collection_if_not_exists(
                collection_name=effective_collection_name,
                vector_size=settings.embedding_dim
            )

            # 2. Extract text
            logger.info(f"[{doc_id}] Extracting text from PDF: {filename}")
            pages_data = extract_text_from_pdf(pdf_bytes)
            if not pages_data:
                logger.warning(f"[{doc_id}] No text extracted from {filename}. Skipping further processing.")
                return doc_id # Or raise DocumentProcessingError("No text could be extracted from the PDF.")

            # 3. Chunk text
            logger.info(f"[{doc_id}] Chunking text for {filename}")
            chunks = chunk_text(
                pages_data,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            if not chunks:
                logger.warning(f"[{doc_id}] No chunks generated from {filename}. Skipping embedding and indexing.")
                return doc_id # Or raise

            # 4. Embed chunks
            logger.info(f"[{doc_id}] Generating embeddings for {len(chunks)} chunks from {filename}")
            chunk_texts_to_embed = [chunk["text"] for chunk in chunks]
            embeddings = embedding_service.embed_texts(chunk_texts_to_embed)
            if len(embeddings) != len(chunks): # Should not happen if embed_texts is robust
                raise DocumentProcessingError("Mismatch between number of chunks and generated embeddings.")

            # 5. Prepare document metadata & Upsert to Vector Store
            logger.info(f"[{doc_id}] Upserting {len(chunks)} chunks to vector store for {filename}")
            document_metadata = {
                "document_id": doc_id,
                "title": filename,
                "author": author, # Will be None if not provided
            }
            vector_store_service.upsert_chunks(
                collection_name=effective_collection_name,
                chunks_data=chunks,
                embeddings=embeddings,
                document_metadata=document_metadata
            )

            logger.info(f"[{doc_id}] Successfully processed and indexed document: {filename}")
            return doc_id

        except PDFParsingError as e:
            logger.error(f"[{doc_id}] PDF parsing failed for {filename}: {e}")
            raise DocumentProcessingError(f"PDF parsing failed for {filename}: {e}") from e
        except EmbeddingError as e:
            logger.error(f"[{doc_id}] Embedding failed for {filename}: {e}")
            raise DocumentProcessingError(f"Embedding failed for {filename}: {e}") from e
        except VectorStoreError as e:
            logger.error(f"[{doc_id}] Vector store operation failed for {filename}: {e}")
            raise DocumentProcessingError(f"Vector store operation failed for {filename}: {e}") from e
        except Exception as e:
            logger.error(f"[{doc_id}] An unexpected error occurred processing {filename}: {e}")
            raise DocumentProcessingError(f"Unexpected error processing {filename}: {e}") from e