# Semantic Q/A Based Search Engine API

## Objective

This project provides a RESTful API that allows users to upload PDF documents, indexes them semantically using vector embeddings, and answers natural language questions based on the content of those documents using a Large Language Model (LLM).

## Features

*   **PDF Document Upload:** Upload one or more PDF files via a REST endpoint.
*   **Asynchronous Processing:** Document parsing, chunking, embedding, and indexing happen in the background to prevent API timeouts.
*   **Semantic Indexing:** Extracts text, splits it into manageable chunks, and generates vector embeddings using a locally run Sentence Transformer model (`sentence-transformers/all-MiniLM-L6-v2` by default).
*   **Vector Storage:** Stores text chunks and their embeddings in a Qdrant vector database.
*   **Question Answering:** Accepts natural language queries, retrieves relevant document chunks based on semantic similarity, and uses an external LLM (e.g., GPT-4 via OpenAI API) to synthesize an answer based *only* on the retrieved context.
*   **Source Attribution:** Provides metadata about the source document chunks used to generate the answer.
*   **Dockerized:** All services (API, Qdrant) are containerized using Docker Compose for easy setup and deployment.
*   **Basic API Structure:** Built with FastAPI, providing automatic interactive documentation (Swagger UI).

## Technical Stack

| Component             | Technology                                         | Notes                                     |
| :-------------------- | :------------------------------------------------- | :---------------------------------------- |
| Language              | Python 3.10+                                       |                                           |
| Web Framework         | FastAPI                                            |                                           |
| Vector DB             | Qdrant                                             | Running as a Docker container             |
| PDF Parsing           | PyMuPDF (`fitz`)                                   |                                           |
| Embedding Model       | Sentence Transformers (`sentence-transformers`)      | Runs **locally** (CPU by default)         |
| LLM for Answering     | OpenAI GPT-4 (configurable)                      | Accessed via **API call**                 |
| Document Chunking     | `RecursiveCharacterTextSplitter` (from LangChain)  |                                           |
| Async Task Processing | FastAPI `BackgroundTasks`                          | For PDF processing                        |
| Development Tools     | Docker, Docker Compose, Git                        |                                           |
| Testing               | Pytest, `unittest.mock`, `pytest-asyncio`        | Unit tests for services                   |

## Project Structure

semantic_qa_api/
├── app/ # FastAPI application code
│ ├── api/ # API Routers, Schemas, Dependencies
│ ├── core/ # Config, Exceptions, Logging
│ ├── services/ # Business logic (document processing, embedding, QA, vector store)
│ └── main.py # FastAPI app instance, middleware, lifespan events
├── tests/ # Pytest tests (unit tests implemented for services)
│ ├── unit/
│ └── fixtures/ # Optional: Sample files for testing
├── .env.example # Example environment variables template
├── .gitignore
├── Dockerfile # Defines the API service Docker image
├── docker-compose.yml # Defines services (API, Qdrant) and their orchestration
├── pytest.ini # Pytest configuration (e.g., warning filters)
├── requirements.txt # Python dependencies
└── README.md

## Setup and Running Locally (Docker Recommended)

Using Docker Compose is the recommended way to run the application locally as it handles both the API service and the Qdrant database.

### Prerequisites

*   Git
*   Docker
*   Docker Compose

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/NouredeenM17/semantic-qa-ai-api
    cd semantic_qa_api
    ```

2.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and fill in the necessary values. **Do not commit the `.env` file to Git.**
        *   `OPENAI_API_KEY`: **Required** for the LLM answering service (GPT-4). Get this from your OpenAI account.
        *   `EMBEDDING_MODEL_NAME`: The Hugging Face identifier for the local Sentence Transformer model (default: `sentence-transformers/all-MiniLM-L6-v2`).
        *   `EMBEDDING_DIM`: **Must match** the dimension of the `EMBEDDING_MODEL_NAME` (default: `384` for `all-MiniLM-L6-v2`).
        *   `LLM_MODEL_NAME`: The OpenAI model to use for answering (default: `gpt-4-turbo-preview`).
        *   `QDRANT_HOST`: Should be `qdrant` when running via Docker Compose (this is the service name).
        *   `QDRANT_PORT`: Should be `6333` (Qdrant's gRPC port).
        *   Other variables (chunk size, overlap, LLM temp/tokens) can usually be left as default.

3.  **Build and Run with Docker Compose:**
    *   From the project root directory (`semantic_qa_api/`), run:
        ```bash
        docker-compose build
        ```
        *(This builds the `api` service image. It might take a while the first time, especially when downloading the embedding model.)*
    *   Then, start the services:
        ```bash
        docker-compose up
        ```
        *(Use `docker-compose up -d` to run in detached mode.)*

4.  **Access the API:**
    *   The API should now be running at `http://localhost:8000`.
    *   Interactive API documentation (Swagger UI) is available at `http://localhost:8000/docs`.
    *   The Qdrant REST API (for debugging/inspection) is available at `http://localhost:6334/dashboard` (if mapped in `docker-compose.yml`).

### Alternative Setup (Local Python without Docker API)

This is more complex as you need to manage dependencies and Qdrant manually.

1.  Ensure Python 3.10+ is installed.
2.  Create and activate a virtual environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
3.  Install dependencies: `pip install -r requirements.txt`.
4.  Run Qdrant separately (e.g., using `docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest`).
5.  Set environment variables (e.g., `export OPENAI_API_KEY="your_key"`). Note: `QDRANT_HOST` should likely be `localhost` in this case.
6.  Run the FastAPI app: `uvicorn app.main:app --reload --port 8000`.

## API Usage

Interact with the API via the Swagger UI at `http://localhost:8000/docs` or use tools like `curl`, Postman, or Insomnia.

### 1. Upload Documents

*   **Endpoint:** `POST /api/v1/documents/upload`
*   **Request Type:** `multipart/form-data`
*   **Form Fields:**
    *   `files`: One or more PDF files (`-F "files=@/path/to/your/document1.pdf" -F "files=@/path/to/your/document2.pdf"`).
    *   `author` (Optional): A string for the author's name (`-F "author=Jane Doe"`).
*   **Response:** `202 Accepted` (if files are valid PDFs). The response body indicates which files were accepted for background processing. The actual document IDs are generated asynchronously.

    ```json
    // Example Response (202 Accepted)
    {
      "message": "2 file(s) accepted and queued for background processing.",
      "document_ids": [ // Note: These are filenames of queued files, not final DB IDs
        "document1.pdf",
        "document2.pdf"
      ],
      "failed_files": []
    }
    ```

*   **`curl` Example:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/documents/upload" \
         -F "files=@/path/to/your/doc1.pdf" \
         -F "files=@/path/to/your/doc2.pdf" \
         -F "author=Test Author"
    ```
    *(Wait a short time for background processing to complete before querying.)*

### 2. Query Documents

*   **Endpoint:** `POST /api/v1/query/query`
*   **Request Type:** `application/json`
*   **Request Body:**
    ```json
    {
      "query": "What is the main topic discussed in the documents?",
      "top_k_retrieval": 3, // Optional, default=5
      "score_threshold": 0.7 // Optional, default=None
    }
    ```
*   **Response:** `200 OK` with the answer and sources.

    ```json
    // Example Response (200 OK)
    {
      "answer": "The documents primarily discuss the implementation details of project 'X' and its performance metrics.",
      "sources": [
        {
          "id": "f8a3b... (chunk_id)",
          "document_id": "e2c4d... (doc_id)",
          "title": "doc1.pdf",
          "page_number": 5,
          "score": 0.8912,
          "text_preview": "Section 3.1 details the implementation specifics for the core module of project 'X'. Performance was evaluated using..."
        },
        {
          "id": "a9b1c... (chunk_id)",
          "document_id": "e2c4d... (doc_id)",
          "title": "doc1.pdf",
          "page_number": 8,
          "score": 0.8555,
          "text_preview": "...the performance metrics gathered in Q4 show a significant improvement after the deployment of project 'X' updates..."
        }
      ]
    }
    ```

*   **`curl` Example:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/query/query" \
         -H "Content-Type: application/json" \
         -d '{
               "query": "What were the Q4 performance metrics?",
               "top_k_retrieval": 2
             }'
    ```

## Running Tests

Unit tests for the service layer have been implemented.

1.  Make sure development dependencies are installed (included in `requirements.txt`).
2.  Run pytest from the project root directory:
    ```bash
    pytest
    ```
    Or for more verbose output:
    ```bash
    pytest -v
    ```

### Activate venv: (on windows CMD)
.\qa-ai-venv\Scripts\activate.bat


