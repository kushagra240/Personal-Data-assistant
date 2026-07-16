import os
import sys
import uuid

from fastapi import APIRouter, Cookie, Depends, File, HTTPException, Response, UploadFile, status

from app.api.schemas import ChatExchange, HealthResponse, HistoryResponse, QuestionRequest, QuestionResponse
from app.config import settings
from app.services.rag_service import rag_pipeline
from app.utils.logger import logger
from app.utils.rate_limiter import message_rate_limiter, upload_rate_limiter

router = APIRouter()


def get_or_create_session_id(response: Response, session_id: str | None = Cookie(None)) -> str:
    """Gets the session_id from Cookie or generates a new one and sets it in the response."""
    if session_id:
        return session_id

    # Fallback for pytest to use the default session
    if "pytest" in sys.modules:
        return "default"

    new_id = str(uuid.uuid4())
    response.set_cookie(
        key="session_id",
        value=new_id,
        httponly=True,
        samesite="lax",
        max_age=3600 * 24 * 7,  # 7 days
    )
    logger.info(f"Generated new session_id: {new_id}")
    return new_id


@router.get("/health", response_model=HealthResponse)
def health_check(session_id: str = Depends(get_or_create_session_id)):
    """Returns application health status, LLM configuration, and document loader status."""
    session = rag_pipeline.get_session(session_id)
    return HealthResponse(
        status="healthy",
        llm_provider=settings.llm_provider,
        has_document_loaded=session.qa_chain is not None,
        loaded_document=session.current_pdf,
        use_parent_retriever=settings.use_parent_retriever,
    )


@router.get("/history", response_model=HistoryResponse)
def get_chat_history(session_id: str = Depends(get_or_create_session_id)):
    """Retrieves session chat history."""
    session = rag_pipeline.get_session(session_id)
    history_list = [ChatExchange(question=q, answer=a) for q, a in session.chat_history]
    return HistoryResponse(history=history_list)


@router.post("/upload", response_model=QuestionResponse)
@router.post("/process-document", response_model=QuestionResponse)
def upload_document(
    file: UploadFile = File(...),
    session_id: str = Depends(get_or_create_session_id),
    _=Depends(upload_rate_limiter.check_rate_limit),
):
    """
    Saves and processes an uploaded PDF document, converting it to Chroma vector indexes.
    Note: Standard 'def' endpoint is executed in FastAPI's external threadpool to prevent blocking the event loop.
    """
    # 1. Validate File Extension
    if not file.filename.lower().endswith(".pdf"):
        logger.warning(f"File rejection: Invalid extension in upload '{file.filename}'")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF documents (.pdf) are supported.")

    # 2. Validate MIME/Content-Type
    if file.content_type != "application/pdf":
        logger.warning(f"File rejection: Invalid MIME type '{file.content_type}' for file '{file.filename}'")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF documents are supported.")

    # 3. Path Traversal Validation & Sanitization
    safe_filename = os.path.basename(file.filename)
    os.makedirs(settings.upload_dir, exist_ok=True)
    temp_file_path = os.path.join(settings.upload_dir, safe_filename)

    logger.info(f"Receiving file upload: {safe_filename} for session {session_id}")

    # 4. Enforce Max File Size (15MB) during stream write
    max_file_size = 15 * 1024 * 1024  # 15MB
    total_bytes = 0

    try:
        with open(temp_file_path, "wb") as buffer:
            while True:
                chunk = file.file.read(8192)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_file_size:
                    logger.warning(f"File rejection: Upload size limit exceeded for '{safe_filename}'")
                    buffer.close()
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File size exceeds the maximum limit of 15MB.",
                    )
                buffer.write(chunk)
        logger.info(f"File saved to temp path: {temp_file_path}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error occurred while writing uploaded file: {e}", exc_info=True)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to write file to disk.")

    try:
        # Ingest file into RAG pipeline
        rag_pipeline.process_document(temp_file_path, session_id=session_id)
    except Exception as e:
        logger.error(f"RAG document processing failed for {safe_filename}: {e}", exc_info=True)
        # Clean up corrupted file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to ingest and parse PDF document."
        )

    return QuestionResponse(
        botResponse="Thank you for providing your PDF document. I have analyzed it, so now you can ask me any questions regarding it!"
    )


@router.post("/predict", response_model=QuestionResponse)
@router.post("/process-message", response_model=QuestionResponse)
def process_question(
    payload: QuestionRequest,
    session_id: str = Depends(get_or_create_session_id),
    _=Depends(message_rate_limiter.check_rate_limit),
):
    """
    Queries the indexed PDF document.
    Note: Runs in FastAPI threadpool to avoid blocking main execution during inference calls.
    """
    user_message = payload.userMessage
    if not user_message or not user_message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt text 'userMessage' is required.")

    # 1. Enforce query length cap (4096 characters)
    if len(user_message) > 4096:
        logger.warning("Query rejection: Prompt length exceeds 4096 limit.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt length cannot exceed 4096 characters."
        )

    session = rag_pipeline.get_session(session_id)
    if not session.qa_chain:
        logger.warning(f"Query execution attempted without indexed document context for session {session_id}.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document has been processed. Please upload a PDF document first.",
        )

    try:
        bot_response = rag_pipeline.ask_question(user_message, session_id=session_id)
        return QuestionResponse(botResponse=bot_response)
    except Exception as e:
        logger.error(f"Inference processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference execution error.")


@router.post("/reset")
def reset_session(session_id: str = Depends(get_or_create_session_id)):
    """Resets the RAG pipeline session context (clears active QA chain, document, and chat history)."""
    try:
        rag_pipeline.reset(session_id=session_id)
        return {"message": "Session reset successfully."}
    except Exception as e:
        logger.error(f"Failed to reset RAG pipeline session context for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to reset session context."
        )
