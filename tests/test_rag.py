from unittest.mock import MagicMock, patch

import pytest

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
def test_rag_pipeline_process_document(
    mock_loader, mock_chroma, mock_embeddings, mock_retrieval_qa, tmp_path, monkeypatch
):
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
    pipeline.qa_chain = MagicMock()  # Mock initialized

    with pytest.raises(ValueError) as exc:
        pipeline.ask_question("   ")
    assert "Question prompt cannot be empty." in str(exc.value)


@patch("app.services.rag_service.RetrievalQA")
@patch("app.services.rag_service.HuggingFaceEmbeddings")
@patch("app.services.rag_service.Chroma")
@patch("app.services.rag_service.PyPDFLoader")
def test_rag_pipeline_process_document_mmr(
    mock_loader, mock_chroma, mock_embeddings, mock_retrieval_qa, tmp_path, monkeypatch
):
    """Tests that the QA Retrieval chain is initialized with MMR parameters when configured."""
    from app.config import settings

    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "mock_key")
    monkeypatch.setattr(settings, "retriever_search_type", "mmr")
    monkeypatch.setattr(settings, "retriever_k", 6)
    monkeypatch.setattr(settings, "retriever_lambda_mult", 0.25)

    pipeline = RAGPipeline()
    pipeline.llm_hub = MagicMock()

    # Configure mock document loader
    mock_doc = MagicMock()
    mock_doc.page_content = "This is a mock PDF document page text."
    mock_doc.metadata = {}
    mock_loader.return_value.load.return_value = [mock_doc]

    mock_db = MagicMock()
    mock_chroma.from_documents.return_value = mock_db

    # Create a temporary empty pdf file
    temp_pdf = tmp_path / "mock_path.pdf"
    temp_pdf.write_bytes(b"%PDF-1.4 dummy pdf content")

    # Run pipeline process
    pipeline.process_document(str(temp_pdf))

    # Assert MMR arguments were passed to as_retriever
    mock_db.as_retriever.assert_called_once_with(search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25})


@patch("app.services.rag_service.RetrievalQA")
@patch("app.services.rag_service.HuggingFaceEmbeddings")
@patch("app.services.rag_service.Chroma")
@patch("app.services.rag_service.ParentDocumentRetriever")
@patch("app.services.rag_service.create_kv_docstore")
@patch("app.services.rag_service.LocalFileStore")
@patch("app.services.rag_service.PyPDFLoader")
def test_rag_pipeline_process_document_parent_retriever(
    mock_loader,
    mock_local_file_store,
    mock_create_kv_docstore,
    mock_parent_retriever,
    mock_chroma,
    mock_embeddings,
    mock_retrieval_qa,
    tmp_path,
    monkeypatch,
):
    """Tests the document ingestion workflow with ParentDocumentRetriever enabled."""
    from app.config import settings

    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "mock_key")
    monkeypatch.setattr(settings, "use_parent_retriever", True)
    monkeypatch.setattr(settings, "child_chunk_size", 128)
    monkeypatch.setattr(settings, "child_chunk_overlap", 16)
    monkeypatch.setattr(settings, "parent_chunk_size", 512)
    monkeypatch.setattr(settings, "parent_chunk_overlap", 32)
    monkeypatch.setattr(settings, "retriever_search_type", "similarity")
    monkeypatch.setattr(settings, "retriever_k", 4)

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

    # Assert correct configurations were passed to ParentDocumentRetriever
    mock_parent_retriever.assert_called_once()
    _, kwargs = mock_parent_retriever.call_args
    assert kwargs["search_type"] == "similarity"
    assert kwargs["search_kwargs"] == {"k": 4}

    # Assert retriever was fed with the loaded documents
    mock_parent_retriever.return_value.add_documents.assert_called_once_with([mock_doc])

    assert pipeline.current_pdf == "mock_path.pdf"
    assert pipeline.qa_chain is not None
    assert pipeline.chat_history == []
    assert pipeline.docstore is not None
    assert pipeline.retriever is not None


@patch("app.services.rag_service.RetrievalQA")
@patch("app.services.rag_service.HuggingFaceEmbeddings")
@patch("app.services.rag_service.Chroma")
@patch("app.services.rag_service.ParentDocumentRetriever")
@patch("app.services.rag_service.create_kv_docstore")
@patch("app.services.rag_service.LocalFileStore")
@patch("app.services.rag_service.PyPDFLoader")
def test_rag_pipeline_session_persistence(
    mock_loader,
    mock_local_file_store,
    mock_create_kv_docstore,
    mock_parent_retriever,
    mock_chroma,
    mock_embeddings,
    mock_retrieval_qa,
    tmp_path,
    monkeypatch,
):
    """Tests that RAGPipeline successfully persists session state to disk and restores it."""
    from app.config import settings

    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "mock_key")
    monkeypatch.setattr(settings, "chroma_db_dir", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(settings, "use_parent_retriever", True)

    pipeline = RAGPipeline()
    pipeline.llm_hub = MagicMock()

    # Configure mock loader
    mock_doc = MagicMock()
    mock_doc.page_content = "Vortex content"
    mock_doc.metadata = {}
    mock_loader.return_value.load.return_value = [mock_doc]

    temp_pdf = tmp_path / "persistence_test.pdf"
    temp_pdf.write_bytes(b"%PDF-1.4 dummy pdf content")

    session_id = "test_persistence_session"

    # Ingest document
    pipeline.process_document(str(temp_pdf), session_id=session_id)
    assert pipeline.get_session(session_id).current_pdf == "persistence_test.pdf"

    # Mock chat history addition
    pipeline.get_session(session_id).chat_history = [("Q1", "A1")]
    pipeline._save_session_metadata(session_id)

    # Simulate server restart by clearing in-memory session cache
    pipeline.sessions.pop(session_id)

    # Re-retrieve session, triggering restoration from disk
    restored_session = pipeline.get_session(session_id)

    assert restored_session.current_pdf == "persistence_test.pdf"
    assert restored_session.chat_history == [["Q1", "A1"]]
    assert restored_session.qa_chain is not None

    # Reset should wipe files
    pipeline.reset(session_id)
    import os

    assert not os.path.exists(os.path.join(settings.chroma_db_dir, f"metadata_{session_id}.json"))
