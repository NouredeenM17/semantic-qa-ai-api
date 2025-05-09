
import pytest
from unittest.mock import patch, MagicMock
from openai import APIError # For mocking OpenAI errors
from app.services.qa_service import QAService, LLM_PROVIDER_OPENAI
from app.core.exceptions import EmbeddingError, VectorStoreError
from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService

# --- Fixtures ---
@pytest.fixture
def mock_openai_chat_completions():
    # This fixture specifically mocks the 'chat.completions.create' path
    with patch('app.services.qa_service.OpenAI') as mock_openai_constructor: # <--- CORRECTED PATCH TARGET
        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create = MagicMock()
        mock_openai_constructor.return_value = mock_openai_instance # When QAService calls OpenAI(), it gets this mock_openai_instance
        yield mock_openai_instance.chat.completions.create # Yield the 'create' method mock

@pytest.fixture
def qa_service_openai(mock_openai_chat_completions, monkeypatch): # Relies on the above fixture
    # Ensure settings are configured for OpenAI for this service instance
    monkeypatch.setattr(settings, "llm_provider", LLM_PROVIDER_OPENAI)
    monkeypatch.setattr(settings, "openai_api_key", "fake_key_for_qa_test")
    monkeypatch.setattr(settings, "llm_model_name", "gpt-test")
    monkeypatch.setattr(settings, "llm_temperature", 0.5)
    monkeypatch.setattr(settings, "llm_max_tokens", 100)
    
    service = QAService()
    # The mock_openai_chat_completions fixture already patched OpenAI constructor,
    # so QAService will use the mocked client internally.
    # We pass the mock_create_method so tests can configure its return value.
    return service, mock_openai_chat_completions


@pytest.fixture
def mock_embedding_service_instance():
    mock = MagicMock(spec=EmbeddingService)
    mock.embed_texts.return_value = [[0.1] * settings.embedding_dim] # Single embedding list
    return mock

@pytest.fixture
def mock_vector_store_service_instance():
    mock = MagicMock(spec=VectorStoreService)
    # search_similar_chunks returns a list of dicts
    mock.search_similar_chunks.return_value = [ 
        {"id": "c1", "payload": {"text": "Context one.", "title": "Doc1", "page_number": 1}, "score": 0.9},
        {"id": "c2", "payload": {"text": "Context two.", "title": "Doc2", "page_number": 5}, "score": 0.8},
    ]
    mock.default_collection_name = settings.qdrant_collection_name # Ensure it has this attr
    return mock

# --- Tests for _build_prompt ---
def test_build_prompt_with_context(qa_service_openai):
    service, _ = qa_service_openai
    query = "What is QA?"
    context_chunks = [
        {"payload": {"text": "QA stands for Quality Assurance.", "title": "doc1.pdf", "page_number": 1}},
        {"payload": {"text": "It ensures software quality.", "title": "doc1.pdf", "page_number": 2}},
    ]
    prompt = service._build_prompt(query, context_chunks)

    assert "Query: What is QA?" in prompt
    assert "Context Information:" in prompt
    assert "Source Document: doc1.pdf, Page: 1" in prompt
    assert "Content: QA stands for Quality Assurance." in prompt
    assert "Source Document: doc1.pdf, Page: 2" in prompt
    assert "Content: It ensures software quality." in prompt
    assert "Answer:" in prompt
    assert "based ONLY on the provided context information" in prompt

def test_build_prompt_no_context(qa_service_openai):
    service, _ = qa_service_openai
    query = "General question?"
    prompt = service._build_prompt(query, []) # Empty context
    # Depending on your safeguard in _build_prompt, it might just return the query
    # or a specific prompt. For now, assuming it returns the query as per current code.
    assert query in prompt 
    # If you change _build_prompt to have specific behavior for no context, update this test.
    assert "Context Information:" not in prompt # Or check if it is there but empty

# --- Tests for get_answer_from_llm ---
def test_get_answer_from_llm_success(qa_service_openai):
    service, mock_llm_create_method = qa_service_openai
    query = "What is love?"
    context_chunks = [{"payload": {"text": "Baby don't hurt me.", "title":"song.txt", "page_number":1}}]

    # Configure the mock LLM response
    mock_choice = MagicMock()
    mock_choice.message.content = " Test LLM answer. "
    mock_llm_response = MagicMock()
    mock_llm_response.choices = [mock_choice]
    mock_llm_create_method.return_value = mock_llm_response

    answer = service.get_answer_from_llm(query, context_chunks)

    assert answer == "Test LLM answer."
    # Check that the mock_llm_create_method (OpenAI().chat.completions.create) was called
    mock_llm_create_method.assert_called_once()
    call_args = mock_llm_create_method.call_args
    assert call_args.kwargs["model"] == "gpt-test" # From monkeypatched settings
    assert call_args.kwargs["temperature"] == 0.5
    assert call_args.kwargs["max_tokens"] == 100
    # You can also assert the content of the prompt in call_args.kwargs["messages"]

def test_get_answer_from_llm_api_error(qa_service_openai):
    service, mock_llm_create_method = qa_service_openai
    query = "Error query"
    context_chunks = [{"payload": {"text": "Some context."}}]

    mock_llm_create_method.side_effect = APIError(message="LLM API Down", request=MagicMock(), body=None)

    with pytest.raises(EmbeddingError, match="OpenAI API error: LLM API Down"): # Or LLMError
        service.get_answer_from_llm(query, context_chunks)

def test_get_answer_from_llm_no_context(qa_service_openai):
    service, _ = qa_service_openai
    query = "Query with no context to LLM"
    # Test the safeguard within get_answer_from_llm
    answer = service.get_answer_from_llm(query, [])
    assert answer == "I cannot answer this question as no relevant context was found."


# --- Tests for answer_query (Orchestration) ---
def test_answer_query_success(
    qa_service_openai, 
    mock_embedding_service_instance, 
    mock_vector_store_service_instance
):
    service, mock_llm_create_method = qa_service_openai # LLM part of qa_service
    query = "What is the main idea?"
    
    # Configure LLM mock response (as it's called by answer_query via get_answer_from_llm)
    mock_choice = MagicMock()
    mock_choice.message.content = "The main idea is X."
    mock_llm_response = MagicMock()
    mock_llm_response.choices = [mock_choice]
    mock_llm_create_method.return_value = mock_llm_response

    response = service.answer_query(
        query=query,
        embedding_service=mock_embedding_service_instance,
        vector_store_service=mock_vector_store_service_instance,
        top_k_retrieval=2,
        score_threshold=0.75
    )

    assert response["answer"] == "The main idea is X."
    assert len(response["sources"]) == 2 # From mock_vector_store_service_instance
    assert response["sources"][0]["title"] == "Doc1"
    assert response["sources"][0]["score"] == 0.9
    assert "text_preview" in response["sources"][0]

    mock_embedding_service_instance.embed_texts.assert_called_once_with([query])
    mock_vector_store_service_instance.search_similar_chunks.assert_called_once_with(
        collection_name=settings.qdrant_collection_name, # Or specific if passed
        query_embedding=[0.1] * settings.embedding_dim, # From mock_embedding_service_instance
        top_k=2,
        score_threshold=0.75
    )
    # mock_llm_create_method was already asserted by virtue of being called by get_answer_from_llm

def test_answer_query_no_relevant_chunks(
    qa_service_openai, 
    mock_embedding_service_instance, 
    mock_vector_store_service_instance
):
    service, _ = qa_service_openai
    query = "Obscure query"
    
    # Make vector store return no chunks
    mock_vector_store_service_instance.search_similar_chunks.return_value = []

    response = service.answer_query(
        query, mock_embedding_service_instance, mock_vector_store_service_instance
    )

    assert response["answer"] == "I could not find any relevant documents to answer your question based on the current criteria."
    assert len(response["sources"]) == 0
    # Ensure LLM was not called
    # Access the mock from the fixture: qa_service_openai[1] is mock_llm_create_method
    qa_service_openai[1].assert_not_called() 

def test_answer_query_embedding_failure(
    qa_service_openai, 
    mock_embedding_service_instance, 
    mock_vector_store_service_instance
):
    service, _ = qa_service_openai
    query = "Query causing embedding error"
    mock_embedding_service_instance.embed_texts.side_effect = EmbeddingError("Embedding failed!")

    response = service.answer_query(
        query, mock_embedding_service_instance, mock_vector_store_service_instance
    )
    
    assert "Error: Could not process the query due to an embedding failure." in response["answer"]
    mock_vector_store_service_instance.search_similar_chunks.assert_not_called()
    qa_service_openai[1].assert_not_called() # LLM not called


def test_answer_query_search_failure(
    qa_service_openai, 
    mock_embedding_service_instance, 
    mock_vector_store_service_instance
):
    service, _ = qa_service_openai
    query = "Query causing search error"
    mock_vector_store_service_instance.search_similar_chunks.side_effect = VectorStoreError("Search failed!")

    response = service.answer_query(
        query, mock_embedding_service_instance, mock_vector_store_service_instance
    )
    
    assert "Error: Could not process the query due to a search failure." in response["answer"]
    qa_service_openai[1].assert_not_called() # LLM not called


def test_answer_query_llm_failure(
    qa_service_openai, 
    mock_embedding_service_instance, 
    mock_vector_store_service_instance
):
    service, mock_llm_create_method = qa_service_openai
    query = "Query causing LLM error"
    
    # mock_vector_store_service_instance.search_similar_chunks is already configured to return chunks
    mock_llm_create_method.side_effect = APIError(message="LLM is down!", request=MagicMock(), body=None)

    response = service.answer_query(
        query, mock_embedding_service_instance, mock_vector_store_service_instance
    )
    
    assert "Error: Could not generate an answer due to an LLM failure." in response["answer"]