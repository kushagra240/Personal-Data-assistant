import os
import shutil
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.config import settings
from app.utils.logger import logger
from app.services.rag_service import rag_pipeline
from app.api.schemas import (
    QuestionRequest,
    QuestionResponse,
    HealthResponse,
    HistoryResponse,
    ChatExchange
)

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Returns application health status, LLM configuration, and document loader status."""
    return HealthResponse(
        status="healthy",
        llm_provider=settings.llm_provider,
        has_document_loaded=rag_pipeline.qa_chain is not None,
        loaded_document=rag_pipeline.current_pdf
    )

@router.get("/history", response_model=HistoryResponse)
def get_chat_history():
    """Retrieves session chat history."""
    history_list = [
        ChatExchange(question=q, answer=a)
        for q, a in rag_pipeline.chat_history
    ]
    return HistoryResponse(history=history_list)

@router.post("/upload", response_model=QuestionResponse)
@router.post("/process-document", response_model=QuestionResponse)
def upload_document(file: UploadFile = File(...)):
    """
    Saves and processes an uploaded PDF document, converting it to Chroma vector indexes.
    Note: Standard 'def' endpoint is executed in FastAPI's external threadpool to prevent blocking the event loop.
    """
    if not file.filename.lower().endswith(".pdf"):
        logger.warning(f"File rejection: Invalid extension in upload '{file.filename}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents (.pdf) are supported."
        )

    # Ensure uploads directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)
    temp_file_path = os.path.join(settings.upload_dir, file.filename)

    logger.info(f"Receiving file upload: {file.filename}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved to temp path: {temp_file_path}")
    except Exception as e:
        logger.error(f"Error occurred while writing uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write file to disk: {str(e)}"
        )

    try:
        # Ingest file into RAG pipeline
        rag_pipeline.process_document(temp_file_path)
    except Exception as e:
        logger.error(f"RAG document processing failed for {file.filename}: {e}")
        # Clean up corrupted file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest and parse PDF document: {str(e)}"
        )

    return QuestionResponse(
        botResponse="Thank you for providing your PDF document. I have analyzed it, so now you can ask me any questions regarding it!"
    )

@router.post("/predict", response_model=QuestionResponse)
@router.post("/process-message", response_model=QuestionResponse)
def process_question(payload: QuestionRequest):
    """
    Queries the indexed PDF document.
    Note: Runs in FastAPI threadpool to avoid blocking main execution during inference calls.
    """
    user_message = payload.userMessage
    if not user_message or not user_message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt text 'userMessage' is required."
        )

    if not rag_pipeline.qa_chain:
        logger.warning("Query execution attempted without indexed document context.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document has been processed. Please upload a PDF document first."
        )

    try:
        bot_response = rag_pipeline.ask_question(user_message)
        return QuestionResponse(botResponse=bot_response)
    except Exception as e:
        logger.error(f"Inference processing failed for query '{user_message}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference server error: {str(e)}"
        )
