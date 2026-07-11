import io
import pytest

def test_health_check_initial(client):
    """Tests the /health endpoint before any document has been loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["has_document_loaded"] is False
    assert data["loaded_document"] is None

def test_history_initial(client):
    """Tests the /history endpoint starts empty."""
    response = client.get("/history")
    assert response.status_code == 200
    assert response.json() == {"history": []}

def test_predict_without_document(client):
    """Tests that sending a query without uploading a PDF returns a 400 Bad Request."""
    response = client.post("/predict", json={"userMessage": "What is the document about?"})
    assert response.status_code == 400
    assert "No document has been processed" in response.json()["detail"]

def test_upload_invalid_extension(client):
    """Tests that uploading a non-PDF file returns a 400 Bad Request."""
    files = {"file": ("test.txt", io.BytesIO(b"dummy text"), "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Only PDF documents (.pdf) are supported." in response.json()["detail"]

def test_upload_valid_pdf(client, mock_rag_pipeline):
    """Tests successfully uploading a PDF file."""
    # Simulate processing setting the QA chain
    mock_rag_pipeline.qa_chain = object() # Mock chain object
    mock_rag_pipeline.current_pdf = "sample.pdf"

    files = {"file": ("sample.pdf", io.BytesIO(b"%PDF-1.4 dummy pdf content"), "application/pdf")}
    response = client.post("/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "Thank you for providing your PDF document" in data["botResponse"]
    
    # Verify health endpoint updates
    health_response = client.get("/health")
    assert health_response.json()["has_document_loaded"] is True
    assert health_response.json()["loaded_document"] == "sample.pdf"

def test_predict_success(client, mock_rag_pipeline):
    """Tests successfully querying the loaded document."""
    # Setup mock loaded state
    mock_rag_pipeline.qa_chain = object()
    mock_rag_pipeline.current_pdf = "sample.pdf"
    
    # Simulate ask_question behavior
    mock_rag_pipeline.chat_history = [("What is RAG?", "Retrieval-Augmented Generation")]
    
    response = client.post("/predict", json={"userMessage": "What is RAG?"})
    assert response.status_code == 200
    assert response.json()["botResponse"] == "Mocked response for testing."

    # Test compatibility route
    response_compat = client.post("/process-message", json={"userMessage": "What is RAG?"})
    assert response_compat.status_code == 200
    assert response_compat.json()["botResponse"] == "Mocked response for testing."

def test_chat_history_populated(client, mock_rag_pipeline):
    """Tests that history returns recorded questions and answers."""
    mock_rag_pipeline.chat_history = [
        ("Query 1", "Answer 1"),
        ("Query 2", "Answer 2")
    ]
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert len(data["history"]) == 2
    assert data["history"][0] == {"question": "Query 1", "answer": "Answer 1"}
