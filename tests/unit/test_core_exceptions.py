
import pytest
from app.core.exceptions import (
    DocumentProcessingError,
    PDFParsingError,
    EmbeddingError,
    VectorStoreError,
)

def test_exception_hierarchy():
    """Test that custom exceptions inherit from expected base or Python exceptions."""
    assert issubclass(PDFParsingError, DocumentProcessingError)
    assert issubclass(EmbeddingError, DocumentProcessingError)
    assert issubclass(VectorStoreError, DocumentProcessingError)
    assert issubclass(DocumentProcessingError, Exception)

def test_raise_pdf_parsing_error():
    with pytest.raises(PDFParsingError, match="Test PDF error"):
        raise PDFParsingError("Test PDF error")

def test_raise_embedding_error():
    with pytest.raises(EmbeddingError, match="Test embedding error"):
        raise EmbeddingError("Test embedding error")

def test_raise_vector_store_error():
    with pytest.raises(VectorStoreError, match="Test vector store error"):
        raise VectorStoreError("Test vector store error")

def test_raise_document_processing_error():
    with pytest.raises(DocumentProcessingError, match="Test doc processing error"):
        raise DocumentProcessingError("Test doc processing error")