import pytest
from unittest.mock import patch, MagicMock, call
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse # For simulating Qdrant errors
from qdrant_client.http.models import ScoredPoint

from app.services.vector_store import VectorStoreService
from app.core.exceptions import VectorStoreError
from app.core.config import settings
import uuid

@pytest.fixture
def mock_qdrant_client_constructor():
    with patch('app.services.vector_store.QdrantClient') as mock_constructor:
        mock_instance = MagicMock(spec=QdrantClient) # Use spec for better mocking
        mock_constructor.return_value = mock_instance
        yield mock_constructor, mock_instance # Yield both constructor and instance

@pytest.fixture
def vector_store_service_instance(mock_qdrant_client_constructor):
    # This fixture provides an already initialized service with a mocked client
    _, mock_client_inst = mock_qdrant_client_constructor
    # We pass the mock_client_inst directly to the constructor for tests
    service = VectorStoreService(qdrant_client=mock_client_inst)
    return service, mock_client_inst


def test_vector_store_service_init_with_provided_client(mock_qdrant_client_constructor):
    mock_constructor, mock_instance = mock_qdrant_client_constructor
    service = VectorStoreService(qdrant_client=mock_instance)
    assert service.client == mock_instance
    mock_constructor.assert_not_called() # Constructor not called if client provided

def test_vector_store_service_init_creates_client(mock_qdrant_client_constructor):
    mock_constructor, mock_instance = mock_qdrant_client_constructor
    
    # To test the __init__ path where it creates its own client:
    # We need to let the original __init__ run, so we patch QdrantClient from its location
    with patch('app.services.vector_store.QdrantClient') as new_mock_constructor:
        new_mock_instance = MagicMock(spec=QdrantClient)
        new_mock_instance.health_check = MagicMock(return_value=None) # Mock health_check attribute
        new_mock_constructor.return_value = new_mock_instance
        
        service = VectorStoreService() # Call without providing client
        
        assert service.client == new_mock_instance
        new_mock_constructor.assert_called_once_with(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            prefer_grpc=settings.qdrant_prefer_grpc,
        )
        new_mock_instance.health_check.assert_called_once()


@pytest.mark.asyncio # For async test functions
async def test_initialize_collection_if_not_exists_creates_collection(vector_store_service_instance):
    service, mock_client = vector_store_service_instance
    
    # Simulate collection NOT existing: get_collection raises an error
    # A generic Exception or a specific qdrant_client.http.exceptions.UnexpectedResponse
    # For UnexpectedResponse, you often need to mock the response object it contains.
    # Simpler: mock get_collection to raise a generic error, then check recreate_collection
    mock_client.get_collection.side_effect = Exception("Collection not found")

    await service.initialize_collection_if_not_exists(collection_name="test_collection")

    mock_client.get_collection.assert_called_once_with(collection_name="test_collection")
    mock_client.recreate_collection.assert_called_once_with(
        collection_name="test_collection",
        vectors_config=models.VectorParams(
            size=settings.embedding_dim,
            distance=models.Distance.COSINE
        )
    )

@pytest.mark.asyncio
async def test_initialize_collection_if_not_exists_already_exists(vector_store_service_instance):
    service, mock_client = vector_store_service_instance
    
    # Simulate collection existing: get_collection returns something (doesn't raise error)
    mock_client.get_collection.return_value = MagicMock() # Or some CollectionInfo object

    await service.initialize_collection_if_not_exists(collection_name="test_collection")

    mock_client.get_collection.assert_called_once_with(collection_name="test_collection")
    mock_client.recreate_collection.assert_not_called()


def test_upsert_chunks_success(vector_store_service_instance):
    service, mock_client = vector_store_service_instance
    
    doc_id = str(uuid.uuid4())
    chunks_data = [
        {"text": "chunk1", "page_number": 1, "chunk_index_in_doc": 0},
        {"text": "chunk2", "page_number": 1, "chunk_index_in_doc": 1},
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    document_metadata = {"document_id": doc_id, "title": "Test Doc", "author": "Test Author"}

    # Mock uuid.uuid4() to return predictable IDs for assertion
    with patch('uuid.uuid4', side_effect=[uuid.UUID('11111111-1111-1111-1111-111111111111'),
                                        uuid.UUID('22222222-2222-2222-2222-222222222222')]):
        service.upsert_chunks(
            chunks_data=chunks_data,
            embeddings=embeddings,
            document_metadata=document_metadata,
            collection_name="test_upsert"
        )

    expected_points = [
        models.PointStruct(
            id='11111111-1111-1111-1111-111111111111',
            vector=[0.1, 0.2],
            payload={
                "text": "chunk1", "page_number": 1, "chunk_index_in_doc": 0,
                "document_id": doc_id, "title": "Test Doc", "author": "Test Author"
            }
        ),
        models.PointStruct(
            id='22222222-2222-2222-2222-222222222222',
            vector=[0.3, 0.4],
            payload={
                "text": "chunk2", "page_number": 1, "chunk_index_in_doc": 1,
                "document_id": doc_id, "title": "Test Doc", "author": "Test Author"
            }
        )
    ]
    mock_client.upsert.assert_called_once_with(
        collection_name="test_upsert", points=expected_points, wait=True
    )

def test_upsert_chunks_mismatch_chunks_embeddings(vector_store_service_instance):
    service, _ = vector_store_service_instance
    with pytest.raises(ValueError, match="Number of chunks and embeddings must match."):
        service.upsert_chunks(
            chunks_data=[{"text": "c1"}],
            embeddings=[[0.1], [0.2]],
            document_metadata={"document_id": "d1"}
        )

def test_upsert_chunks_qdrant_error(vector_store_service_instance):
    service, mock_client = vector_store_service_instance
    mock_client.upsert.side_effect = Exception("Qdrant upsert failed")

    with pytest.raises(VectorStoreError, match="Failed to upsert points to Qdrant: Qdrant upsert failed"):
        service.upsert_chunks(
            chunks_data=[{"text": "c1", "page_number": 1, "chunk_index_in_doc": 0}],
            embeddings=[[0.1]],
            document_metadata={"document_id": "d1", "title": "t1"}
        )

def test_search_similar_chunks_success(vector_store_service_instance):
    service, mock_client = vector_store_service_instance
    
    query_embedding = [0.1] * settings.embedding_dim # Dummy embedding
    top_k = 3
    collection_name = "test_search_collection"

    # Mock Qdrant's search response
    mock_search_results = [
        ScoredPoint(id=str(uuid.uuid4()), version=1, score=0.9, payload={"text": "chunk1", "page_number": 1}, vector=None, shard_key=None),
        ScoredPoint(id=str(uuid.uuid4()), version=1, score=0.8, payload={"text": "chunk2", "page_number": 2}, vector=None, shard_key=None),
    ]
    mock_client.search.return_value = mock_search_results

    results = service.search_similar_chunks(
        query_embedding=query_embedding,
        top_k=top_k,
        collection_name=collection_name,
        score_threshold=0.7
    )

    assert len(results) == 2
    assert results[0]["payload"]["text"] == "chunk1"
    assert results[0]["score"] == 0.9
    assert "id" in results[0] # Check that ID is included

    mock_client.search.assert_called_once_with(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        score_threshold=0.7
    )

def test_search_similar_chunks_no_results(vector_store_service_instance):
    service, mock_client = vector_store_service_instance
    query_embedding = [0.2] * settings.embedding_dim
    top_k = 5
    
    mock_client.search.return_value = [] # Qdrant returns an empty list

    results = service.search_similar_chunks(query_embedding, top_k)

    assert len(results) == 0
    mock_client.search.assert_called_once_with(
        collection_name=settings.qdrant_collection_name, # Using default
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        score_threshold=None # Default score_threshold
    )

def test_search_similar_chunks_qdrant_error(vector_store_service_instance):
    service, mock_client = vector_store_service_instance
    query_embedding = [0.3] * settings.embedding_dim
    
    # Simulate a Qdrant client error
    mock_client.search.side_effect = UnexpectedResponse(status_code=500, headers="header", content="Server error", reason_phrase="Server Error")

    with pytest.raises(VectorStoreError, match="Failed to search Qdrant:"):
        service.search_similar_chunks(query_embedding, top_k=3)