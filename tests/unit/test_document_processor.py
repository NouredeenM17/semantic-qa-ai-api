# tests/unit/test_document_processor.py
import pytest
from unittest.mock import patch, MagicMock # For mocking
import fitz # PyMuPDF, for its error class
from app.services.document_processor import (
    extract_text_from_pdf,
    chunk_text,
    # process_and_index_pdf, # We'll test this later
)
from app.core.exceptions import PDFParsingError, DocumentProcessingError
from app.core.config import settings # For default chunk_size/overlap

# --- Tests for extract_text_from_pdf ---

@pytest.fixture
def sample_pdf_bytes():
    # Create a minimal valid PDF in memory for testing
    # This avoids needing a file, but is more complex to set up accurately.
    # For simplicity, we'll assume you have a sample.pdf or mock fitz.open entirely.
    # If using a file:
    try:
        with open("tests/fixtures/basic-text.pdf", "rb") as f:
            return f.read()
    except FileNotFoundError:
        pytest.skip("tests/fixtures/basic-text.pdf not found, skipping PDF read test.")
        return None


def test_extract_text_from_pdf_success(sample_pdf_bytes):
    if not sample_pdf_bytes:
        pytest.skip("No sample PDF bytes to test.")

    pages = extract_text_from_pdf(sample_pdf_bytes)
    assert len(pages) > 0 # Or the exact number of pages in your sample
    assert isinstance(pages[0], tuple)
    assert isinstance(pages[0][0], int) # Page number
    assert isinstance(pages[0][1], str) # Page text
    # Add more specific assertions based on your sample.pdf content
    # e.g., assert "Page 1 content" in pages[0][1]

@patch('fitz.open') # Mock the fitz.open function
def test_extract_text_from_pdf_parsing_error(mock_fitz_open):
    # Configure the mock to raise a RuntimeError when a method like load_page is called
    mock_doc = MagicMock()
    mock_doc.is_encrypted = False # Assume not encrypted for this specific parsing error test
    mock_doc.__len__.return_value = 1 # Simulate having 1 page
    mock_doc.load_page.side_effect = RuntimeError("mocked PyMuPDF runtime error")
    mock_fitz_open.return_value = mock_doc

    pdf_bytes = b"dummy_pdf_content"
    # The error message in PDFParsingError will now reflect the RuntimeError
    with pytest.raises(PDFParsingError, match="Failed to parse PDF \(PyMuPDF runtime error\): mocked PyMuPDF runtime error"):
        extract_text_from_pdf(pdf_bytes)

@patch('fitz.open')
def test_extract_text_from_pdf_encrypted_error(mock_fitz_open):
    mock_doc = MagicMock()
    mock_doc.is_encrypted = True
    mock_doc.authenticate.return_value = 0 # Simulate failed authentication (returns 0)
    mock_fitz_open.return_value = mock_doc

    pdf_bytes = b"dummy_encrypted_pdf_content"
    with pytest.raises(PDFParsingError, match="PDF is password-protected and requires a password to open."):
        extract_text_from_pdf(pdf_bytes)

@patch('fitz.open')
def test_extract_text_from_pdf_no_text_extracted(mock_fitz_open):
    mock_page = MagicMock()
    mock_page.get_text.return_value = "" # Page has no text
    mock_doc = MagicMock()
    mock_doc.is_encrypted = False
    mock_doc.__len__.return_value = 1 # One page
    mock_doc.load_page.return_value = mock_page
    mock_fitz_open.return_value = mock_doc

    pages = extract_text_from_pdf(b"dummy_image_pdf_content")
    assert len(pages) == 0
    # Check logs for warning if you implement that

# --- Tests for chunk_text ---

@pytest.fixture
def sample_pages_data():
    return [
        (1, "This is the first page. It has some text."),
        (2, "Second page here. A bit more content to make it longer than one chunk hopefully."),
        (3, "Short third page."),
    ]

# Mocking RecursiveCharacterTextSplitter directly
@patch('app.services.document_processor.RecursiveCharacterTextSplitter')
def test_chunk_text_basic(mock_splitter_class, sample_pages_data):
    # Configure the mock splitter instance
    mock_splitter_instance = MagicMock()
    # Simulate how split_documents would return LangchainDocument-like objects
    def mock_split_documents(docs):
        results = []
        chunk_content_map = {
            "This is the first page. It has some text.": ["Chunk 1.1"],
            "Second page here. A bit more content to make it longer than one chunk hopefully.": ["Chunk 2.1", "Chunk 2.2 overlap"],
            "Short third page.": ["Chunk 3.1"]
        }
        for doc in docs:
            page_num = doc.metadata["page_number"]
            for content in chunk_content_map.get(doc.page_content, []):
                mock_chunk_doc = MagicMock()
                mock_chunk_doc.page_content = content
                mock_chunk_doc.metadata = {"page_number": page_num}
                results.append(mock_chunk_doc)
        return results

    mock_splitter_instance.split_documents.side_effect = mock_split_documents
    mock_splitter_class.return_value = mock_splitter_instance # Constructor returns our mock

    chunks = chunk_text(sample_pages_data, chunk_size=10, chunk_overlap=2) # size/overlap are for mock config

    assert len(chunks) == 4 # Based on mock_split_documents
    assert chunks[0]["text"] == "Chunk 1.1"
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_index_in_doc"] == 0

    assert chunks[1]["text"] == "Chunk 2.1"
    assert chunks[1]["page_number"] == 2
    assert chunks[1]["chunk_index_in_doc"] == 1
    
    assert chunks[2]["text"] == "Chunk 2.2 overlap"
    assert chunks[2]["page_number"] == 2
    assert chunks[2]["chunk_index_in_doc"] == 2

    assert chunks[3]["text"] == "Chunk 3.1"
    assert chunks[3]["page_number"] == 3
    assert chunks[3]["chunk_index_in_doc"] == 3

    mock_splitter_class.assert_called_once_with(
        chunk_size=10, chunk_overlap=2, length_function=len, is_separator_regex=False
    )

def test_chunk_text_empty_input():
    chunks = chunk_text([])
    assert len(chunks) == 0

def test_chunk_text_empty_page_content():
    # This test verifies that pages with no text are skipped before calling the splitter
    pages_data = [(1, "  \n  ")] # Page with only whitespace
    # No need to mock the splitter if it's not supposed to be called
    chunks = chunk_text(pages_data)
    assert len(chunks) == 0