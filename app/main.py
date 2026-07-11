import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.endpoints import router as api_router
from app.config import settings
from app.utils.logger import logger

# Initialize FastAPI application
app = FastAPI(
    title="Production RAG Chatbot Portfolio",
    description="FastAPI & LangChain RAG pipeline migrated from IBM Watsonx to Hugging Face & Google Gemini",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route endpoints directly at root level to maintain 100% compatibility with original Flask schema
app.include_router(api_router)

# Resolve templates and static directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(base_dir, "static")
templates_dir = os.path.join(base_dir, "templates")

# Ensure static files directory exists and mount it
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_index():
    """Serves the main single page application frontend."""
    index_path = os.path.join(templates_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    logger.warning(f"Frontend template index.html not found at: {index_path}")
    return {
        "message": "FastAPI RAG Backend is running. Please add 'templates/index.html' to access the user interface.",
        "endpoints": {
            "swagger": "/docs",
            "health": "/health",
            "history": "/history"
        }
    }

# FastAPI Startup event handler
@app.on_event("startup")
def on_startup():
    logger.info("FastAPI Server starting up...")
    logger.info(f"Active Configuration: LLM_PROVIDER={settings.llm_provider}, HF_MODEL={settings.hf_model_id}")
