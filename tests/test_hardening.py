import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request, HTTPException
from app.utils.rate_limiter import RateLimiter

def test_rate_limiter_limit():
    """Tests that the RateLimiter correctly triggers a 429 status code on exceeding request limit."""
    limiter = RateLimiter(requests_limit=2, window_seconds=10)
    
    # Mock FastAPI request object
    mock_request = MagicMock(spec=Request)
    mock_request.client.host = "127.0.0.1"
    
    # First 2 requests should pass
    limiter.check_rate_limit(mock_request)
    limiter.check_rate_limit(mock_request)
    
    # 3rd request should exceed limit and raise 429
    with pytest.raises(HTTPException) as exc:
        limiter.check_rate_limit(mock_request)
    
    assert exc.value.status_code == 429
    assert exc.value.detail == "Rate limit exceeded. Please try again later."

def test_predict_prompt_length_cap(client, mock_rag_pipeline):
    """Tests that a prompt exceeding 4096 characters is rejected with 400 Bad Request."""
    mock_rag_pipeline.qa_chain = object()
    mock_rag_pipeline.current_pdf = "test.pdf"
    
    long_prompt = "a" * 4097
    response = client.post("/predict", json={"userMessage": long_prompt})
    
    assert response.status_code == 400
    assert "Prompt length cannot exceed 4096 characters." in response.json()["detail"]

def test_predict_empty_prompt(client, mock_rag_pipeline):
    """Tests that empty or whitespace-only prompts are rejected with 400."""
    mock_rag_pipeline.qa_chain = object()
    mock_rag_pipeline.current_pdf = "test.pdf"
    
    response = client.post("/predict", json={"userMessage": "   "})
    assert response.status_code == 400
    assert "Prompt text 'userMessage' is required." in response.json()["detail"]

def test_upload_invalid_mime_type(client):
    """Tests uploading a file with a .pdf extension but invalid MIME type."""
    files = {"file": ("sample.pdf", io.BytesIO(b"%PDF-1.4 dummy content"), "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Only PDF documents are supported." in response.json()["detail"]

def test_upload_file_size_exceeded(client):
    """Tests that files exceeding the 15MB file size limit are rejected with 413 Payload Too Large."""
    files = {"file": ("large.pdf", io.BytesIO(b"%PDF-1.4 dummy context"), "application/pdf")}
    
    # Patch SpooledTemporaryFile.read to return a 16MB chunk on the first read call
    from tempfile import SpooledTemporaryFile
    
    first_read = [True]
    def mock_read(self, *args, **kwargs):
        if first_read[0]:
            first_read[0] = False
            return b"a" * (16 * 1024 * 1024)
        return b""
        
    with patch.object(SpooledTemporaryFile, "read", mock_read):
        response = client.post("/upload", files=files)
    
    assert response.status_code == 413
    assert "File size exceeds the maximum limit of 15MB." in response.json()["detail"]

def test_error_masking_predict(client, mock_rag_pipeline):
    """Tests that internal exceptions during prompt prediction are masked as general 500 errors."""
    mock_rag_pipeline.qa_chain = object()
    mock_rag_pipeline.current_pdf = "test.pdf"
    
    # Mock ask_question to raise raw internal exception
    mock_rag_pipeline.ask_question.side_effect = RuntimeError("Database connection lost or secret key corrupted")
    
    response = client.post("/predict", json={"userMessage": "What is the key revenue?"})
    
    assert response.status_code == 500
    # The response detail should not leak "Database connection lost or secret key corrupted"
    assert "Inference execution error." in response.json()["detail"]
    assert "Database connection lost" not in response.json()["detail"]

def test_error_masking_upload(client, mock_rag_pipeline):
    """Tests that internal exceptions during document ingestion are masked as general 500 errors."""
    mock_rag_pipeline.qa_chain = object()
    mock_rag_pipeline.current_pdf = None
    
    # Mock process_document to raise raw internal exception
    mock_rag_pipeline.process_document.side_effect = ValueError("Fatal indexing collision in Chroma index tree")
    
    files = {"file": ("sample.pdf", io.BytesIO(b"%PDF-1.4 dummy content"), "application/pdf")}
    response = client.post("/upload", files=files)
    
    assert response.status_code == 500
    assert "Failed to ingest and parse PDF document." in response.json()["detail"]
    assert "indexing collision" not in response.json()["detail"]

def test_error_masking_reset(client, mock_rag_pipeline):
    """Tests that internal exceptions during reset session are masked as general 500 errors."""
    # Mock reset to raise raw internal exception
    mock_rag_pipeline.reset.side_effect = OSError("Access denied: cannot remove directory database files")
    
    response = client.post("/reset")
    
    assert response.status_code == 500
    assert "Failed to reset session context." in response.json()["detail"]
    assert "Access denied" not in response.json()["detail"]
