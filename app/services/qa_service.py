# app/services/qa_service.py
import logging
from typing import List, Dict, Optional

from openai import OpenAI, APIError
from app.core.config import settings
from app.core.exceptions import EmbeddingError # Re-using for LLM errors for now, or create new LLMError
from app.services.embedding_service import EmbeddingService # For query embedding
from app.services.vector_store import VectorStoreService # For searching

logger = logging.getLogger(__name__)

# Constants for LLM Providers (can be expanded)
LLM_PROVIDER_OPENAI = "openai"
LLM_PROVIDER_GEMINI = "gemini"
LLM_PROVIDER_MOCK = "mock"

class QAService:
    def __init__(self):
        self.llm_provider = settings.llm_provider.lower()
        
        if self.llm_provider == LLM_PROVIDER_OPENAI:
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY must be set for OpenAI provider.")
            try:
                self.openai_client = OpenAI(api_key=settings.openai_api_key)
                logger.info(f"QAService initialized with OpenAI provider, model: {settings.llm_model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client for QAService: {e}")
                raise EmbeddingError(f"Failed to initialize OpenAI client for QAService: {e}") # Or LLMError
        elif self.llm_provider == LLM_PROVIDER_GEMINI:
            # Placeholder: Initialize Gemini client here
            # if not settings.gemini_api_key:
            #     raise ValueError("GEMINI_API_KEY must be set for Gemini provider.")
            # self.gemini_client = ...
            logger.warning("Gemini provider selected but not yet fully implemented in QAService.")
            raise NotImplementedError("Gemini LLM provider is not yet implemented.")
        elif self.llm_provider == LLM_PROVIDER_MOCK:
            pass
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

        self.llm_model_name = settings.llm_model_name
        self.llm_temperature = settings.llm_temperature
        self.llm_max_tokens = settings.llm_max_tokens


    def _build_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Builds the prompt for the LLM with context.

        Args:
            query: The user's query.
            context_chunks: A list of context chunk dictionaries, each expected
                            to have 'payload' (with 'text', 'page_number', 'title').
        """
        if not context_chunks:
            # This case should ideally be handled before calling _build_prompt,
            # but as a safeguard:
            return query # Or a specific prompt asking to answer generally

        context_str = "\n\n---\n\n".join(
            f"Source Document: {chunk.get('payload', {}).get('title', 'N/A')}, Page: {chunk.get('payload', {}).get('page_number', 'N/A')}\nContent: {chunk.get('payload', {}).get('text', '')}"
            for chunk in context_chunks
        )

        prompt = f"""You are a helpful AI assistant. Answer the following query based ONLY on the provided context information.
            If the answer cannot be found in the context, state "I cannot answer this question based on the provided context."
            Do not use any external knowledge or information not present in the context.
            Context Information:
            {context_str}
            Query: {query}
            Answer:"""

        return prompt


    def get_answer_from_llm(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Gets an answer from the configured LLM based on the query and context.

        Args:
            query: The user's query.
            context_chunks: List of context chunks retrieved from the vector store.

        Returns:
            The LLM's generated answer.

        Raises:
            EmbeddingError (or LLMError): If the LLM call fails.
        """
        if not context_chunks: # Should be handled by the orchestrator
            logger.warning("get_answer_from_llm called with no context_chunks.")
            return "I cannot answer this question as no relevant context was found."

        prompt = self._build_prompt(query, context_chunks)
        logger.debug(f"Generated LLM Prompt:\n{prompt}")

        try:
            if self.llm_provider == LLM_PROVIDER_OPENAI:
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[
                        # {"role": "system", "content": "You are a helpful AI assistant..."}, # System prompt can be part of the user prompt or separate
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_tokens,
                )
                answer = response.choices[0].message.content.strip()
            elif self.llm_provider == LLM_PROVIDER_GEMINI:
                # Placeholder for Gemini API call
                # response = self.gemini_client.generate_content(...)
                # answer = ...
                raise NotImplementedError("Gemini LLM provider call is not yet implemented.")
            elif self.llm_provider == LLM_PROVIDER_MOCK:
                answer = prompt + " THIS IS A MOCK LLM RESPONSE."
            else:
                # Should have been caught in __init__
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

            logger.info(f"LLM ({self.llm_model_name}) generated answer successfully.")
            return answer

        except APIError as e:
            logger.error(f"OpenAI API error during LLM call: {e.message}")
            raise EmbeddingError(f"OpenAI API error: {e.message}") # Or LLMError
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM call: {e}")
            raise EmbeddingError(f"Failed to get answer from LLM: {e}") # Or LLMError

    
    def answer_query(
        self,
        query: str,
        embedding_service: EmbeddingService, # Dependency
        vector_store_service: VectorStoreService, # Dependency
        collection_name: Optional[str] = None, # To pass to vector_store
        top_k_retrieval: int = 5, # How many chunks to retrieve
        score_threshold: Optional[float] = 0.7 # Similarity threshold for retrieval
    ) -> Dict:
        """
        Orchestrates the process of answering a query:
        1. Embeds the query.
        2. Searches for relevant chunks in the vector store.
        3. If chunks are found, passes them and the query to an LLM to generate an answer.
        4. Formats the response including the answer and source metadata.

        Args:
            query: The user's natural language query.
            embedding_service: Instance of EmbeddingService.
            vector_store_service: Instance of VectorStoreService.
            collection_name: Optional name of the Qdrant collection.
            top_k_retrieval: Number of chunks to retrieve from vector store.
            score_threshold: Minimum similarity score for retrieved chunks.

        Returns:
            A dictionary containing the 'answer' and 'sources'.
            Example: {'answer': 'The answer is...', 
                      'sources': [{'title': 'doc1.pdf', 'page_number': 3, 'score': 0.88, 'text_preview': '...'}, ...]}
        """
        logger.info(f"Received query: '{query}'")
        effective_collection_name = collection_name or vector_store_service.default_collection_name

        # 1. Embed the query
        try:
            query_embedding = embedding_service.embed_texts([query])[0]
            logger.debug("Query embedded successfully.")
        except Exception as e: # Catching generic exception from embedding_service
            logger.error(f"Failed to embed query '{query}': {e}")
            # Or re-raise as a specific QueryProcessingError
            return {"answer": "Error: Could not process the query due to an embedding failure.", "sources": []}

        # 2. Search for relevant chunks
        try:
            relevant_chunks = vector_store_service.search_similar_chunks(
                collection_name=effective_collection_name,
                query_embedding=query_embedding,
                top_k=top_k_retrieval,
                score_threshold=score_threshold
            )
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks.")
        except Exception as e: # Catching generic exception from vector_store_service
            logger.error(f"Failed to search for relevant chunks for query '{query}': {e}")
            return {"answer": "Error: Could not process the query due to a search failure.", "sources": []}

        # 3. Check if relevant chunks were found
        if not relevant_chunks:
            logger.info(f"No relevant documents found for query: '{query}' with threshold {score_threshold}")
            return {"answer": "I could not find any relevant documents to answer your question based on the current criteria.", "sources": []}

        # 4. Use LLM to generate answer from selected chunks
        try:
            llm_answer = self.get_answer_from_llm(query, relevant_chunks)
            logger.info(f"LLM generated answer for query: '{query}'")
        except Exception as e: # Catching generic exception from get_answer_from_llm
            logger.error(f"Failed to get answer from LLM for query '{query}': {e}")
            return {"answer": "Error: Could not generate an answer due to an LLM failure.", "sources": []}


        # 5. Format the response with source metadata
        sources_metadata = []
        for chunk in relevant_chunks:
            payload = chunk.get("payload", {})
            # Add a short preview of the context text for better source attribution
            text_preview = payload.get("text", "")[:150] + "..." if payload.get("text") else "N/A"
            sources_metadata.append({
                "id": chunk.get("id"), # Chunk ID from Qdrant
                "document_id": payload.get("document_id"),
                "title": payload.get("title", "N/A"),
                "page_number": payload.get("page_number"),
                "score": chunk.get("score"), # Similarity score
                "text_preview": text_preview
            })
        
        logger.info(f"Successfully processed query: '{query}'")
        return {"answer": llm_answer, "sources": sources_metadata}