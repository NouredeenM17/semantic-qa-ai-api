
import logging
from typing import List, Dict, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse # For specific Qdrant errors
from app.core.config import settings
from app.core.exceptions import VectorStoreError
import uuid # For generating chunk IDs

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self, qdrant_client: Optional[QdrantClient] = None):
        if qdrant_client:
            self.client = qdrant_client
        else:
            # Initialize a new client if one isn't provided
            # This allows for dependency injection for testing or managed clients
            try:
                self.client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    prefer_grpc=settings.qdrant_prefer_grpc,
                    # timeout=10 # Optional: set timeout for requests
                )
                #self.client.health_check() # Verifies connection
                logger.info(f"VectorStoreService initialized and connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                raise VectorStoreError(f"Could not connect to Qdrant: {e}")
        
        self.default_collection_name = settings.qdrant_collection_name


    async def initialize_collection_if_not_exists(
        self,
        collection_name: Optional[str] = None,
        vector_size: int = settings.embedding_dim,
        distance_metric: models.Distance = models.Distance.COSINE
    ):
        """
        Creates the Qdrant collection if it doesn't already exist.
        This method is async to align with FastAPI's async nature if called during startup.
        """
        col_name = collection_name or self.default_collection_name
        try:
            # Check if collection exists
            # The way to check existence can vary slightly with client versions
            # A common way is to try to get collection info and catch an exception
            try:
                self.client.get_collection(collection_name=col_name)
                logger.info(f"Collection '{col_name}' already exists.")
                return
            except (UnexpectedResponse, Exception) as e: # Catch specific "not found" or general error
                # A more robust check might look for a specific "not found" status code (e.g. 404)
                # For now, we assume if get_collection fails, it might not exist.
                pass # Collection might not exist, proceed to create

            logger.info(f"Collection '{col_name}' does not exist. Creating...")
            self.client.recreate_collection( # or create_collection if you are sure it doesn't exist
                collection_name=col_name,
                vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance_metric
                    )
                # Potentially add HNSW or other indexing parameters here for production
                # hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100)
            )
            logger.info(f"Successfully created collection '{col_name}' with vector size {vector_size} and distance {distance_metric}.")
        except Exception as e:
            logger.error(f"Failed to initialize or create collection '{col_name}': {e}")
            raise VectorStoreError(f"Failed to initialize Qdrant collection '{col_name}': {e}")

    def upsert_chunks(
        self,
        chunks_data: List[Dict], # List of {'text': ..., 'page_number': ..., 'chunk_index_in_doc': ...}
        embeddings: List[List[float]],
        document_metadata: Dict, # {'document_id': ..., 'title': ..., 'author': ...}
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Upserts document chunks and their embeddings into Qdrant.

        Args:
            collection_name: Name of the Qdrant collection.
            chunks_data: List of dictionaries, each containing chunk text and metadata.
            embeddings: List of vector embeddings corresponding to the chunks.
            document_metadata: Dictionary containing metadata for the parent document.

        Raises:
            VectorStoreError: If the upsert operation fails.
        """
        col_name = collection_name or self.default_collection_name
        if len(chunks_data) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")
        if not chunks_data:
            logger.info("No chunks to upsert.")
            return

        points_to_upsert: List[models.PointStruct] = []
        for i, chunk_info in enumerate(chunks_data):
            chunk_id = str(uuid.uuid4()) # Unique ID for each chunk
            payload = {
                "text": chunk_info["text"],
                "page_number": chunk_info["page_number"],
                "chunk_index_in_doc": chunk_info["chunk_index_in_doc"],
                "document_id": document_metadata["document_id"],
                "title": document_metadata.get("title", "N/A"), # PDF filename
                "author": document_metadata.get("author"), # Optional
                # "user_id": authenticated_user_id
            }
            # Filter out None values from payload as Qdrant might not like them for certain field types
            payload = {k: v for k, v in payload.items() if v is not None}


            points_to_upsert.append(
                models.PointStruct(
                    id=chunk_id,
                    vector=embeddings[i],
                    payload=payload,
                )
            )

        try:
            # Upsert points in batches if necessary, Qdrant client handles batching well.
            self.client.upsert(collection_name=col_name, points=points_to_upsert, wait=True)
            logger.info(f"Successfully upserted {len(points_to_upsert)} points to collection '{col_name}'.")
        except Exception as e:
            logger.error(f"Failed to upsert points to Qdrant collection '{col_name}': {e}")
            raise VectorStoreError(f"Failed to upsert points to Qdrant: {e}")


    def search_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None # Optional similarity score threshold
    ) -> List[Dict]:
        """
        Searches Qdrant for the top_k most similar chunks to the query_embedding.

        Args:
            collection_name: Name of the Qdrant collection.
            query_embedding: The vector embedding of the query.
            top_k: The maximum number of similar chunks to retrieve.
            score_threshold: Optional. Minimum similarity score for a chunk to be included.
                             (For Cosine similarity, higher is better, typically 0.7-1.0)

        Returns:
            A list of dictionaries, where each dictionary contains the payload
            of a relevant chunk and its similarity score.
            Example: [{'payload': {...}, 'score': 0.85}, ...]

        Raises:
            VectorStoreError: If the search operation fails.
        """
        col_name = collection_name or self.default_collection_name
        try:
            # query_filter = models.Filter(
            #     must=[
            #         models.FieldCondition(
            #             key="user_id", # The key in your payload
            #             match=models.MatchValue(value=authenticated_user_id)
            #         )
            #     ]
            # )

            search_results = self.client.search(
                collection_name=col_name,
                query_vector=query_embedding,
                # query_filter=query_filter,
                limit=top_k,
                with_payload=True,  # We need the payload (text, metadata)
                with_vectors=False, # We don't usually need the vector itself in the result
                score_threshold=score_threshold # Filter by similarity score if provided
            )
            
            # search_results is a list of ScoredPoint objects
            # ScoredPoint(id=..., version=..., score=..., payload=..., vector=..., shard_key=...)
            
            results = []
            for hit in search_results:
                results.append({
                    "id": str(hit.id), # Good to have the point ID
                    "payload": hit.payload,
                    "score": hit.score
                })
            
            logger.info(f"Found {len(results)} similar chunks in '{col_name}' for the query (top_k={top_k}, score_threshold={score_threshold}).")
            return results

        except Exception as e:
            logger.error(f"Failed to search Qdrant collection '{col_name}': {e}")
            raise VectorStoreError(f"Failed to search Qdrant: {e}")

# Global instance or dependency injection:
# For simplicity in early stages, a global instance can be okay,
# but for testability, dependency injection is better.
# We'll handle instantiation in app.main or via FastAPI deps later.