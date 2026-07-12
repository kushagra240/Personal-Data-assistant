import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    debug: bool = Field(default=True, validation_alias="DEBUG")
    
    # LLM Settings: 'huggingface', 'gemini', or 'watsonx'
    llm_provider: str = Field(default="huggingface", validation_alias="LLM_PROVIDER")
    
    # Hugging Face Settings
    huggingfacehub_api_token: str = Field(default="", validation_alias="HUGGINGFACEHUB_API_TOKEN")
    hf_model_id: str = Field(default="meta-llama/Llama-3.1-8B-Instruct", validation_alias="HF_MODEL_ID")
    
    # Gemini Settings
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_model_id: str = Field(default="gemini-3.5-flash", validation_alias="GEMINI_MODEL_ID")

    # Watsonx Settings
    watsonx_apikey: str = Field(default="", validation_alias="WATSONX_APIKEY")
    watsonx_project_id: str = Field(default="skills-network", validation_alias="WATSONX_PROJECT_ID")
    watsonx_url: str = Field(default="https://us-south.ml.cloud.ibm.com", validation_alias="WATSONX_URL")
    watsonx_model_id: str = Field(default="meta-llama/llama-4-maverick-17b-128e-instruct-fp8", validation_alias="WATSONX_MODEL_ID")
    watsonx_max_new_tokens: int = Field(default=256, validation_alias="WATSONX_MAX_NEW_TOKENS")
    watsonx_temperature: float = Field(default=0.1, validation_alias="WATSONX_TEMPERATURE")
    
    # CORS Settings
    cors_allowed_origins: str = Field(default="http://localhost:8000,http://127.0.0.1:8000", validation_alias="CORS_ALLOWED_ORIGINS")
    
    # Vector DB / Embeddings Settings
    embedding_model_id: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL_ID")
    chroma_db_dir: str = Field(default="./data/chroma_db", validation_alias="CHROMA_DB_DIR")
    upload_dir: str = Field(default="./data/uploads", validation_alias="UPLOAD_DIR")

    # Retriever Settings
    retriever_search_type: str = Field(default="similarity", validation_alias="RETRIEVER_SEARCH_TYPE")
    retriever_k: int = Field(default=3, validation_alias="RETRIEVER_K")
    retriever_lambda_mult: float = Field(default=0.25, validation_alias="RETRIEVER_LAMBDA_MULT")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Create a singleton settings object
settings = Settings()
