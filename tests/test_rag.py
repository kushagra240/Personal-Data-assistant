import os
import pytest
from unittest.mock import MagicMock, patch
from app.services.rag_service import RAGPipeline

def test_rag_pipeline_init_invalid_provider(monkeypatch):
    """Tests that initializing the RAG pipeline with an invalid LLM provider raises ValueError."""
    from app.config import settings
    monkeypatch.setattr(settings, "llm_provider", "invalid_provider")
    
    pipeline = RAGPipeline()
    with pytest.raises(ValueError) as exc:
        pipeline.init_llm()
    assert "Invalid LLM_PROVIDER" in str(exc.value)

@patch("app.services.rag_service.RetrievalQA")
@patch("app.services.rag_service.HuggingFaceEmbeddings")
@patch("app.services.rag_service.Chroma")
@patch("app.services.rag_service.PyPDFLoader")
def test_rag_pipeline_process_document(mock_loader, mock_chroma, mock_embeddings, mock_retrieval_qa, tmp_path, monkeypatch):
    """Tests the document parsing, splitting, embedding, and QA chain generation workflow."""
    # Mock settings
    from app.config import settings
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "mock_key")
    
    pipeline = RAGPipeline()
    pipeline.llm_hub = MagicMock()
    
    # Configure mock document loader
    mock_doc = MagicMock()
    mock_doc.page_content = "This is a mock PDF document page text."
    mock_doc.metadata = {}
    mock_loader.return_value.load.return_value = [mock_doc]
    
    # Create a temporary empty pdf file
    temp_pdf = tmp_path / "mock_path.pdf"
    temp_pdf.write_bytes(b"%PDF-1.4 dummy pdf content")
    
    # Run pipeline process
    pipeline.process_document(str(temp_pdf))
    
    # Assert loader and Chroma were invoked
    mock_loader.assert_called_once_with(str(temp_pdf))
    mock_chroma.from_documents.assert_called_once()
    assert pipeline.current_pdf == "mock_path.pdf"
    assert pipeline.qa_chain is not None
    assert pipeline.chat_history == []

def test_ask_question_no_document_error():
    """Tests that querying when no document is loaded raises a ValueError."""
    pipeline = RAGPipeline()
    with pytest.raises(ValueError) as exc:
        pipeline.ask_question("What is RAG?")
    assert "No PDF document loaded." in str(exc.value)

def test_ask_question_empty_prompt():
    """Tests that querying with an empty prompt raises a ValueError."""
    pipeline = RAGPipeline()
    pipeline.qa_chain = MagicMock() # Mock initialized
    
    with pytest.raises(ValueError) as exc:
        pipeline.ask_question("   ")
    assert "Question prompt cannot be empty." in str(exc.value)
