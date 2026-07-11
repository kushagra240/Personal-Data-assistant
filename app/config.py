import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    debug: bool = Field(default=True, validation_alias="DEBUG")
    
    # LLM Settings: 'huggingface' or 'gemini'
    llm_provider: str = Field(default="huggingface", validation_alias="LLM_PROVIDER")
    
    # Hugging Face Settings
    huggingfacehub_api_token: str = Field(default="", validation_alias="HUGGINGFACEHUB_API_TOKEN")
    hf_model_id: str = Field(default="meta-llama/Llama-3.1-8B-Instruct", validation_alias="HF_MODEL_ID")
    
    # Gemini Settings
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_model_id: str = Field(default="gemini-2.5-flash", validation_alias="GEMINI_MODEL_ID")
    
    # Vector DB / Embeddings Settings
    embedding_model_id: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL_ID")
    chroma_db_dir: str = Field(default="./data/chroma_db", validation_alias="CHROMA_DB_DIR")
    upload_dir: str = Field(default="./data/uploads", validation_alias="UPLOAD_DIR")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Create a singleton settings object
settings = Settings()
