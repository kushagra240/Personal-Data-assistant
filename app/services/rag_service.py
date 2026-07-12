import os
import torch
from app.config import settings
from app.utils.logger import logger

# Imports for LangChain RAG pipeline
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class RAGPipeline:
    def __init__(self):
        self.llm_hub = None
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.chat_history = []
        self.current_pdf = None

    def init_llm(self):
        """Initializes the embeddings model and choice of LLM provider."""
        logger.info(f"Initializing LLM with provider: {settings.llm_provider}")
        
        # Check GPU availability for embeddings computation
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device '{device}' for HuggingFace embeddings.")
        
        # Initialize embeddings model (local MiniLM)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_id,
            model_kwargs={"device": device}
        )
        logger.info(f"Embeddings model '{settings.embedding_model_id}' successfully initialized.")

        # Initialize LLM based on provider
        if settings.llm_provider == "huggingface":
            # Guard for missing/placeholder API token
            if not settings.huggingfacehub_api_token or settings.huggingfacehub_api_token == "your_huggingface_api_token_here":
                logger.warning("HUGGINGFACEHUB_API_TOKEN is a placeholder or not set. Activating DEMO MODE with Fake LLM.")
                from langchain_core.language_models.fake import FakeListLLM
                self.llm_hub = FakeListLLM(responses=[
                    "[DEMO MODE] Hugging Face API token is not set. The RAG pipeline processed the PDF successfully and searched the Chroma vector index. To get real AI responses, configure a valid token in the .env file."
                ] * 100)
                return
            
            from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
            
            logger.info("Initializing HuggingFaceEndpoint LLM...")
            base_llm = HuggingFaceEndpoint(
                repo_id=settings.hf_model_id,
                task="text-generation",
                huggingfacehub_api_token=settings.huggingfacehub_api_token,
                temperature=0.1,
                max_new_tokens=600,
            )
            self.llm_hub = ChatHuggingFace(llm=base_llm)
            logger.info(f"ChatHuggingFace LLM initialized with model '{settings.hf_model_id}'.")
            
        elif settings.llm_provider == "gemini":
            # Guard for missing/placeholder API key
            if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
                logger.warning("GEMINI_API_KEY is a placeholder or not set. Activating DEMO MODE with Fake LLM.")
                from langchain_core.language_models.fake import FakeListLLM
                self.llm_hub = FakeListLLM(responses=[
                    "[DEMO MODE] Gemini API key is not set. The RAG pipeline processed the PDF successfully and searched the Chroma vector index. To get real AI responses, configure a valid token in the .env file."
                ] * 100)
                return
                
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            logger.info("Initializing ChatGoogleGenerativeAI LLM...")
            self.llm_hub = ChatGoogleGenerativeAI(
                model=settings.gemini_model_id,
                google_api_key=settings.gemini_api_key,
                temperature=0.1,
                max_output_tokens=600,
            )
            logger.info(f"ChatGoogleGenerativeAI LLM initialized with model '{settings.gemini_model_id}'.")
        elif settings.llm_provider == "watsonx":
            # Guard for missing/placeholder API key
            if not settings.watsonx_apikey or settings.watsonx_apikey == "your_watsonx_apikey_here":
                logger.warning("WATSONX_APIKEY is a placeholder or not set. Activating DEMO MODE with Fake LLM.")
                from langchain_core.language_models.fake import FakeListLLM
                self.llm_hub = FakeListLLM(responses=[
                    "[DEMO MODE] Watsonx API key is not set. The RAG pipeline processed the PDF successfully and searched the Chroma vector index. To get real AI responses, configure a valid API key in the .env file."
                ] * 100)
                return
                
            from langchain_ibm import WatsonxLLM
            
            # WatsonxLLM checks WATSONX_APIKEY env var internally
            os.environ["WATSONX_APIKEY"] = settings.watsonx_apikey
            
            logger.info("Initializing WatsonxLLM...")
            model_parameters = {
                "max_new_tokens": settings.watsonx_max_new_tokens,
                "temperature": settings.watsonx_temperature,
            }
            self.llm_hub = WatsonxLLM(
                model_id=settings.watsonx_model_id,
                url=settings.watsonx_url,
                project_id=settings.watsonx_project_id,
                params=model_parameters
            )
            logger.info(f"WatsonxLLM initialized with model '{settings.watsonx_model_id}'.")
        else:
            raise ValueError(f"Invalid LLM_PROVIDER '{settings.llm_provider}'. Supported: 'huggingface', 'gemini', 'watsonx'.")

    def process_document(self, file_path: str):
        """Processes a PDF document, splits it, embeds chunks, and builds a QA Retrieval Chain."""
        if not self.embeddings:
            self.init_llm()
            
        logger.info(f"Starting document processing for: {file_path}")
        
        # Load PDF pages
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at path: {file_path}")
            
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Successfully loaded PDF. Page count: {len(documents)}")

        # Split document text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=64
        )
        texts = text_splitter.split_documents(documents)
        logger.info(f"Document split into {len(texts)} chunks.")

        # Create Chroma DB vector store and persist
        os.makedirs(settings.chroma_db_dir, exist_ok=True)
        logger.info(f"Loading chunks into Chroma DB at: {settings.chroma_db_dir}")
        self.vector_store = Chroma.from_documents(
            texts, 
            embedding=self.embeddings,
            persist_directory=settings.chroma_db_dir
        )
        logger.info("Chroma DB initialization complete.")

        # Build QA Retrieval Chain
        search_kwargs = {"k": settings.retriever_k}
        if settings.retriever_search_type == "mmr":
            search_kwargs["lambda_mult"] = settings.retriever_lambda_mult
            
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_hub,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type=settings.retriever_search_type,
                search_kwargs=search_kwargs
            ),
            return_source_documents=False,
            input_key="question"
        )
        self.current_pdf = os.path.basename(file_path)
        # Clear chat history for the new document
        self.chat_history = []
        logger.info("RetrievalQA chain generated and ready.")

    def ask_question(self, question: str) -> str:
        """Queries the retrieval QA chain and appends exchange to chat history."""
        if not self.qa_chain:
            raise ValueError("No PDF document loaded. Please upload a PDF file first.")
        
        if not question or not question.strip():
            raise ValueError("Question prompt cannot be empty.")
            
        logger.info(f"Invoking RAG pipeline QA chain for question: '{question}'")
        output = self.qa_chain.invoke(
            {"question": question, "chat_history": self.chat_history}
        )
        answer = output["result"]
        
        # Update session chat history
        self.chat_history.append((question, answer))
        logger.info("Updated chat history.")
        return answer.strip()

# Singleton instance of RAG pipeline
rag_pipeline = RAGPipeline()
