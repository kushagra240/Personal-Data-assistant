import os
import sys
from unittest.mock import MagicMock

import pytest

# Add the project root directory to the sys.path to allow imports of 'app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import app components
from app.config import settings


@pytest.fixture(autouse=True)
def mock_settings_dirs(tmp_path):
    """Overrides Chroma and upload directories to point to a temporary test folder."""
    settings.chroma_db_dir = str(tmp_path / "chroma_db")
    settings.upload_dir = str(tmp_path / "uploads")
    yield


@pytest.fixture
def mock_rag_pipeline(monkeypatch):
    """Mocks the RAGPipeline singleton instance to run tests without contacting upstream LLM APIs."""
    from app.services.rag_service import rag_pipeline

    mock_init = MagicMock()
    mock_process = MagicMock()
    mock_ask = MagicMock(return_value="Mocked response for testing.")

    monkeypatch.setattr(rag_pipeline, "init_llm", mock_init)
    monkeypatch.setattr(rag_pipeline, "process_document", mock_process)
    monkeypatch.setattr(rag_pipeline, "ask_question", mock_ask)

    # Pre-populate some properties to simulate loaded state when needed
    rag_pipeline.current_pdf = None
    rag_pipeline.qa_chain = None
    rag_pipeline.chat_history = []

    return rag_pipeline


@pytest.fixture
def client(mock_rag_pipeline):
    """Provides a TestClient for querying the FastAPI endpoints."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
