class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    pass

class PDFParsingError(DocumentProcessingError):
    """Exception raised for errors during PDF parsing."""
    pass

class EmbeddingError(DocumentProcessingError):
    """Exception raised for errors during text embedding."""
    pass

class VectorStoreError(DocumentProcessingError):
    """Exception raised for errors interacting with the vector store."""
    pass