
import pytest
from unittest.mock import patch, MagicMock
from openai import APIError

from app.services.embedding_service import EmbeddingService
from app.core.exceptions import EmbeddingError
from app.core.config import settings # Import the settings instance

@pytest.fixture
def mock_openai_client_constructor(): # Renamed to reflect it mocks the constructor
    with patch('app.services.embedding_service.OpenAI') as mock_client_constructor:
        mock_client_instance = MagicMock()
        mock_client_constructor.return_value = mock_client_instance
        yield mock_client_constructor, mock_client_instance

def test_embedding_service_init_success(mock_openai_client_constructor, monkeypatch): # Add monkeypatch fixture
    mock_constructor, mock_instance = mock_openai_client_constructor
    
    # Use monkeypatch to temporarily set the attribute on the settings instance
    monkeypatch.setattr(settings, "openai_api_key", "test_key_success")
    
    service = EmbeddingService() # EmbeddingService will read the patched settings.openai_api_key
    assert service.client is not None
    assert service.model_name == settings.embedding_model_name
    mock_constructor.assert_called_once_with(api_key="test_key_success")


def test_embedding_service_init_no_api_key(monkeypatch):
    # Temporarily set the API key to an empty string
    monkeypatch.setattr(settings, "openai_api_key", "")
    
    with pytest.raises(ValueError, match="OPENAI_API_KEY must be set"):
        EmbeddingService()
    # monkeypatch automatically restores the original value after the test

def test_embed_texts_success(mock_openai_client_constructor, monkeypatch):
    mock_constructor, mock_client_instance = mock_openai_client_constructor
    monkeypatch.setattr(settings, "openai_api_key", "test_key_embed")

    mock_embedding_data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]
    mock_response = MagicMock()
    mock_response.data = mock_embedding_data
    mock_client_instance.embeddings.create.return_value = mock_response

    service = EmbeddingService()
    texts = ["hello", "world"]
    embeddings = service.embed_texts(texts)

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    mock_client_instance.embeddings.create.assert_called_once_with(
        input=texts, model=settings.embedding_model_name
    )

def test_embed_texts_empty_input(mock_openai_client_constructor, monkeypatch):
    mock_constructor, mock_client_instance = mock_openai_client_constructor
    monkeypatch.setattr(settings, "openai_api_key", "test_key_empty")
    
    service = EmbeddingService()
    embeddings = service.embed_texts([])
    assert embeddings == []
    mock_client_instance.embeddings.create.assert_not_called()

def test_embed_texts_openai_api_error(mock_openai_client_constructor, monkeypatch):
    mock_constructor, mock_client_instance = mock_openai_client_constructor
    monkeypatch.setattr(settings, "openai_api_key", "test_key_api_error")

    mock_client_instance.embeddings.create.side_effect = APIError(
        message="Mocked OpenAI API Error", request=MagicMock(), body=None # ensure request is a mock
    )
    
    service = EmbeddingService()
    with pytest.raises(EmbeddingError, match="OpenAI API error: Mocked OpenAI API Error"):
        service.embed_texts(["test text"])

def test_embed_texts_unexpected_error(mock_openai_client_constructor, monkeypatch):
    mock_constructor, mock_client_instance = mock_openai_client_constructor
    monkeypatch.setattr(settings, "openai_api_key", "test_key_unexpected")

    mock_client_instance.embeddings.create.side_effect = Exception("Unexpected generic error")

    service = EmbeddingService()
    with pytest.raises(EmbeddingError, match="Failed to embed texts: Unexpected generic error"):
        service.embed_texts(["test text"])