#  Semantic Q/A Based Search Engine API

##  Objective

Develop a RESTful API that:

1. Allows users to upload documents (PDF format).
2. Indexes the documents semantically using vector embeddings (stored in **Qdrant**).
3. Accepts natural language queries and performs semantic similarity search.
4. Leverages a **Large Language Model (LLM)** to answer questions based on the most relevant document chunks.
5. Handles gracefully when no relevant results are found.

---

##  Deliverables

- REST API with endpoints:
  - `POST /upload`: Upload and index PDF documents.
  - `POST /query`: Accept natural language queries and return context-based answers.
- Backend service that:
  - Extracts and chunks text from PDFs.
  - Embeds and stores document chunks in Qdrant.
  - Retrieves relevant content and generates answers using LLM.
- Proper logging, error handling, and test coverage.
- `README.md` with setup instructions and API usage.
- Postman collection or Swagger documentation.

---

## 🔧 Technical Stack

| Component             | Technology                          |
|----------------------|--------------------------------------|
| Language             | Python                               |
| Web Framework        | FastAPI or Flask                     |
| Vector DB            | Qdrant                               |
| PDF Parsing          | PyMuPDF or PDFMiner                  |
| Embedding Model      | OpenAI  or Hugging Face              |
| LLM for Answering    | OpenAI GPT-4, Claude, etc            |
| Document Chunking    | Recursive Character Splitter (LangChain or custom) |
| Dev Tools            | Docker, Git, Postman                 |
| Testing              | Pytest                               |

---

##  Functional Details

### 🔹 `POST /upload` – Upload Endpoint

- **Input**: One or multiple PDF files.
- **Steps**:
  - Extract text from PDFs.
  - Define a chunking strategy example: Chunk text (e.g., 500–1000 tokens/chunk).
  - Generate vector embeddings for each chunk.
  - Store chunks and metadata in Qdrant. (metadata includes id, title, author, page number)
- **Response**: Document ID(s), status message.

### 🔹 `POST /query` – Search Endpoint

- **Input**: Natural language query (JSON).
- **Steps**:
  - Embed query using same embedding model.
  - Search Qdrant for top-K relevant chunks.
  - Use LLM to generate answer from selected chunks.
- **Response**:
  - Answer string.
  - Source document metadata.
  - Graceful fallback: “No relevant documents found.”

---

##  Backend Process Flow

1. Extract text from PDFs.
2. Chunk text using overlap (to preserve context).
3. Embed chunks via LLM embedding model.
4. Index chunks in Qdrant with metadata.
5. Embed incoming query.
6. Retrieve similar chunks from Qdrant.
7. Pass chunks + query into LLM.
8. Return synthesized answer.

---

##  Evaluation Criteria

| Area         | Criteria                                                   |
|--------------|------------------------------------------------------------|
| Functionality | API endpoints behave as expected                          |
| Accuracy      | Relevant content retrieved and used in LLM-generated answers |
| Robustness    | Handles corrupt PDFs, empty queries, and large files      |
| Scalability   | Handles multiple concurrent documents and queries         |
| Performance   | Responds in under a minute (accounting for LLM latency)   |
| Code Quality  | Clean, modular, annotated code with proper error handling |
| Documentation | Clear setup guide and endpoint documentation              |

---

## Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI API](https://platform.openai.com/docs/)
- [LangChain](https://docs.langchain.com/)
- Use PDFs for testing both English and Arabic texts (Arabic is optional)

---

## Optional Enhancements

- Add API authentication (JWT/API Keys).
- Multi-language support for PDFs. (Arabic recommended)
- Rate limiting and query history logging.
- Testing 80%+ unit/integration test coverage  

---

Good luck!
