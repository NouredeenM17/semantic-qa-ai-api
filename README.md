# semantic-qa-ai-api

## Objective

This project provides a RESTful API that allows users to upload PDF documents, indexes them semantically using vector embeddings, and answers natural language questions based on the content of those documents using a Large Language Model (LLM).

Tech stack used: FastAPI, Qdrant, OpenAI Embeddings & GPT-4 (initially), PyMuPDF, LangChain Chunking, pytest, Docker Development Environment.

## Technical Stack

| Component             | Technology                                         | Notes                                     |
| :-------------------- | :------------------------------------------------- | :---------------------------------------- |
| Language              | Python 3.10+                                       |                                           |
| Web Framework         | FastAPI                                            |                                           |
| Vector DB             | Qdrant                                             | Running as a Docker container             |
| PDF Parsing           | PyMuPDF (`fitz`)                                   |                                           |
| Embedding Model       | Sentence Transformers (`sentence-transformers`)    | Runs **locally** (CPU by default)         |
| LLM for Answering     | OpenAI GPT-4 (configurable)                        | Accessed via **API call**                 |
| Document Chunking     | `RecursiveCharacterTextSplitter` (from LangChain)  |                                           |
| Async Task Processing | FastAPI `BackgroundTasks`                          | For PDF processing                        |
| Development Tools     | Docker, Docker Compose, Git                        |                                           |
| Testing               | Pytest, `unittest.mock`, `pytest-asyncio`          | Unit tests for services                   |


# Activate venv: (on windows CMD)
.\qa-ai-venv\Scripts\activate.bat


# Run unit tests with
bash: pytest

