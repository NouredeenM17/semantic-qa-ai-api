
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from app.services.embedding_service import EmbeddingService
from app.core.exceptions import EmbeddingError
from app.core.config import settings

# --- Fixtures ---

@pytest.fixture
def mock_sentence_transformer_constructor():
    """Mocks the SentenceTransformer class constructor and its instance methods."""
    # Patch where SentenceTransformer is looked up by EmbeddingService
    with patch('app.services.embedding_service.SentenceTransformer') as mock_st_constructor:
        mock_st_instance = MagicMock()
        mock_st_instance.encode.return_value = [
            np.array([0.1, 0.2, 0.3]), 
            np.array([0.4, 0.5, 0.6])
        ]
        mock_st_constructor.return_value = mock_st_instance
        yield mock_st_constructor, mock_st_instance

# --- Tests for __init__ ---

def test_embedding_service_init_success(mock_sentence_transformer_constructor, monkeypatch):
    mock_st_constructor, mock_st_instance = mock_sentence_transformer_constructor
    
    # Ensure settings reflect a model name for the test
    monkeypatch.setattr(settings, "embedding_model_name", "cpu-test-model")

    service = EmbeddingService()

    assert service.local_model == mock_st_instance
    assert service.model_name == "cpu-test-model"
    # Verify SentenceTransformer was called with 'cpu' device
    mock_st_constructor.assert_called_once_with("cpu-test-model", device='cpu')

def test_embedding_service_init_model_load_failure(mock_sentence_transformer_constructor, monkeypatch):
    mock_st_constructor, _ = mock_sentence_transformer_constructor

    # Simulate SentenceTransformer constructor raising an error
    mock_st_constructor.side_effect = Exception("Failed to download/load model")
    
    monkeypatch.setattr(settings, "embedding_model_name", "bad-model-name-cpu")

    with pytest.raises(EmbeddingError, match="Failed to load local model: Failed to download/load model"):
        EmbeddingService()

@patch('app.services.embedding_service.SentenceTransformer', None) # Simulate library not installed
def test_embedding_service_init_library_not_installed(monkeypatch):
    # No need to mock torch if it's not directly used in the simplified __init__ for device detection
    monkeypatch.setattr(settings, "embedding_model_name", "any-model-cpu")
    with pytest.raises(ImportError, match="SentenceTransformers library is required for local embeddings but not installed."): # Adjusted match
        EmbeddingService()


# --- Tests for embed_texts ---

def test_embed_texts_success(mock_sentence_transformer_constructor, monkeypatch):
    _ , mock_st_instance = mock_sentence_transformer_constructor # We only need the instance
    monkeypatch.setattr(settings, "embedding_model_name", "test-model-embed-cpu")

    service = EmbeddingService() # Initialize the service (uses mocked ST)
    
    texts_to_embed = ["hello cpu", "another cpu text"]
    # This should match what mock_st_instance.encode is configured to return in the fixture
    expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] 

    embeddings = service.embed_texts(texts_to_embed)

    assert embeddings == expected_embeddings
    mock_st_instance.encode.assert_called_once_with(texts_to_embed, convert_to_tensor=False)

def test_embed_texts_empty_input(mock_sentence_transformer_constructor, monkeypatch):
    _ , mock_st_instance = mock_sentence_transformer_constructor
    monkeypatch.setattr(settings, "embedding_model_name", "test-model-empty-cpu")

    service = EmbeddingService()
    
    embeddings = service.embed_texts([])

    assert embeddings == []
    mock_st_instance.encode.assert_not_called()

def test_embed_texts_encode_failure(mock_sentence_transformer_constructor, monkeypatch):
    _ , mock_st_instance = mock_sentence_transformer_constructor
    monkeypatch.setattr(settings, "embedding_model_name", "test-model-fail-encode-cpu")

    mock_st_instance.encode.side_effect = Exception("Internal ST encode error on CPU")

    service = EmbeddingService()
    texts_to_embed = ["some text for cpu error"]

    with pytest.raises(EmbeddingError, match="Failed to embed texts: Internal ST encode error on CPU"):
        service.embed_texts(texts_to_embed)