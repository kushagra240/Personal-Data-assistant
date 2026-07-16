import os

import torch

# Imports for LangChain RAG pipeline
from langchain.chains import RetrievalQA
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.utils.logger import logger


class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.qa_chain = None
        self.chat_history = []
        self.current_pdf = None
        self.docstore = None
        self.retriever = None
        self.vector_store = None


class RAGPipeline:
    def __init__(self):
        self.llm_hub = None
        self.embeddings = None
        self.sessions = {}

    def get_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id)
        return self.sessions[session_id]

    @property
    def qa_chain(self):
        return self.get_session("default").qa_chain

    @qa_chain.setter
    def qa_chain(self, value):
        self.get_session("default").qa_chain = value

    @property
    def chat_history(self):
        return self.get_session("default").chat_history

    @chat_history.setter
    def chat_history(self, value):
        self.get_session("default").chat_history = value

    @property
    def current_pdf(self):
        return self.get_session("default").current_pdf

    @current_pdf.setter
    def current_pdf(self, value):
        self.get_session("default").current_pdf = value

    @property
    def docstore(self):
        return self.get_session("default").docstore

    @docstore.setter
    def docstore(self, value):
        self.get_session("default").docstore = value

    @property
    def retriever(self):
        return self.get_session("default").retriever

    @retriever.setter
    def retriever(self, value):
        self.get_session("default").retriever = value

    @property
    def vector_store(self):
        return self.get_session("default").vector_store

    @vector_store.setter
    def vector_store(self, value):
        self.get_session("default").vector_store = value

    def init_llm(self):
        """Initializes the embeddings model and choice of LLM provider."""
        logger.info(f"Initializing LLM with provider: {settings.llm_provider}")

        # Check GPU availability for embeddings computation
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device '{device}' for HuggingFace embeddings.")

        # Initialize embeddings model (local MiniLM)
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_id, model_kwargs={"device": device})
        logger.info(f"Embeddings model '{settings.embedding_model_id}' successfully initialized.")

        # Initialize LLM based on provider
        if settings.llm_provider == "huggingface":
            # Guard for missing/placeholder API token
            if (
                not settings.huggingfacehub_api_token
                or settings.huggingfacehub_api_token == "your_huggingface_api_token_here"
            ):
                logger.warning(
                    "HUGGINGFACEHUB_API_TOKEN is a placeholder or not set. Activating DEMO MODE with Fake LLM."
                )
                from langchain_core.language_models.fake import FakeListLLM

                self.llm_hub = FakeListLLM(
                    responses=[
                        "[DEMO MODE] Hugging Face API token is not set. The RAG pipeline processed the PDF successfully and searched the Chroma vector index. To get real AI responses, configure a valid token in the .env file."
                    ]
                    * 100
                )
                return

            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

            logger.info("Initializing HuggingFaceEndpoint LLM...")
            base_llm = HuggingFaceEndpoint(
                repo_id=settings.hf_model_id,
                task="text-generation",
                huggingfacehub_api_token=settings.huggingfacehub_api_token,
                temperature=settings.llm_temperature,
                max_new_tokens=settings.llm_max_new_tokens,
            )
            self.llm_hub = ChatHuggingFace(llm=base_llm)
            logger.info(f"ChatHuggingFace LLM initialized with model '{settings.hf_model_id}'.")

        elif settings.llm_provider == "gemini":
            # Guard for missing/placeholder API key
            if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
                logger.warning("GEMINI_API_KEY is a placeholder or not set. Activating DEMO MODE with Fake LLM.")
                from langchain_core.language_models.fake import FakeListLLM

                self.llm_hub = FakeListLLM(
                    responses=[
                        "[DEMO MODE] Gemini API key is not set. The RAG pipeline processed the PDF successfully and searched the Chroma vector index. To get real AI responses, configure a valid token in the .env file."
                    ]
                    * 100
                )
                return

            from langchain_google_genai import ChatGoogleGenerativeAI

            logger.info("Initializing ChatGoogleGenerativeAI LLM...")
            self.llm_hub = ChatGoogleGenerativeAI(
                model=settings.gemini_model_id,
                google_api_key=settings.gemini_api_key,
                temperature=settings.llm_temperature,
                max_output_tokens=settings.llm_max_new_tokens,
            )
            logger.info(f"ChatGoogleGenerativeAI LLM initialized with model '{settings.gemini_model_id}'.")
        else:
            raise ValueError(f"Invalid LLM_PROVIDER '{settings.llm_provider}'. Supported: 'huggingface', 'gemini'.")

    def process_document(self, file_path: str, session_id: str = "default"):
        """Processes a PDF document, splits it, embeds chunks, and builds a QA Retrieval Chain."""
        if not self.embeddings:
            self.init_llm()

        session = self.get_session(session_id)
        logger.info(f"Starting document processing for session {session_id}, file: {file_path}")

        # Clear any existing Chroma DB collections/directories for this session to prevent context-leakage
        if session.vector_store:
            try:
                session.vector_store.delete_collection()
                logger.info(f"Deleted existing Chroma collection for session {session_id} to clear context.")
            except Exception as e:
                logger.warning(f"Could not delete collection before document ingestion for session {session_id}: {e}")
            session.vector_store = None
        else:
            collection_name = f"vortex_rag_{session_id}"
            try:
                # Direct cleanup of existing collection with the same name if any
                temp_vs = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=settings.chroma_db_dir,
                )
                temp_vs.delete_collection()
                logger.info(f"Deleted pre-existing Chroma collection '{collection_name}' to clean up.")
            except Exception:
                pass

        # Load PDF pages
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at path: {file_path}")

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Successfully loaded PDF. Page count: {len(documents)}")

        # Create Chroma DB vector store and build retriever
        os.makedirs(settings.chroma_db_dir, exist_ok=True)

        search_kwargs = {"k": settings.retriever_k}
        if settings.retriever_search_type == "mmr":
            search_kwargs["lambda_mult"] = settings.retriever_lambda_mult

        collection_name = f"vortex_rag_{session_id}"

        if settings.use_parent_retriever:
            logger.info(f"Using ParentDocumentRetriever for advanced context retrieval in session {session_id}.")

            session.docstore = InMemoryStore()

            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.child_chunk_size, chunk_overlap=settings.child_chunk_overlap
            )
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.parent_chunk_size, chunk_overlap=settings.parent_chunk_overlap
            )

            session.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_db_dir,
            )

            session.retriever = ParentDocumentRetriever(
                vectorstore=session.vector_store,
                docstore=session.docstore,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_type=settings.retriever_search_type,
                search_kwargs=search_kwargs,
            )

            # ParentDocumentRetriever automatically splits the original documents and populates stores
            session.retriever.add_documents(documents)
            logger.info(f"ParentDocumentRetriever setup and document ingestion complete for session {session_id}.")
        else:
            logger.info(f"Using standard document retriever in session {session_id}.")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.parent_chunk_size, chunk_overlap=settings.parent_chunk_overlap
            )
            texts = text_splitter.split_documents(documents)
            logger.info(f"Document split into {len(texts)} chunks.")

            logger.info(f"Loading chunks into Chroma DB collection '{collection_name}' at: {settings.chroma_db_dir}")
            session.vector_store = Chroma.from_documents(
                texts,
                embedding=self.embeddings,
                persist_directory=settings.chroma_db_dir,
                collection_name=collection_name,
            )
            session.retriever = session.vector_store.as_retriever(
                search_type=settings.retriever_search_type, search_kwargs=search_kwargs
            )
            logger.info(f"Standard retriever setup complete for session {session_id}.")

        # Build QA Retrieval Chain
        prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Provide a detailed, comprehensive, and well-structured response. Organize your answer with clear points, bullet points, or sections if appropriate.

Context:
{context}

Question: {question}
Helpful Answer:"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        session.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_hub,
            chain_type="stuff",
            retriever=session.retriever,
            return_source_documents=False,
            input_key="question",
            chain_type_kwargs={"prompt": PROMPT},
        )
        session.current_pdf = os.path.basename(file_path)
        # Clear chat history for the new document
        session.chat_history = []
        logger.info(f"RetrievalQA chain generated and ready for session {session_id}.")

    def ask_question(self, question: str, session_id: str = "default") -> str:
        """Queries the retrieval QA chain and appends exchange to chat history."""
        session = self.get_session(session_id)
        if not session.qa_chain:
            raise ValueError("No PDF document loaded. Please upload a PDF file first.")

        if not question or not question.strip():
            raise ValueError("Question prompt cannot be empty.")

        logger.info(f"Invoking RAG pipeline QA chain for session {session_id}, question: '{question}'")
        output = session.qa_chain.invoke({"question": question, "chat_history": session.chat_history})
        answer = output["result"]

        # Update session chat history
        session.chat_history.append((question, answer))
        logger.info(f"Updated chat history for session {session_id}.")
        return answer.strip()

    def reset(self, session_id: str = "default"):
        """Resets the pipeline context, clearing the active QA chain, document, and chat history."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # Clean up Chroma collection for this session
            if session.vector_store:
                try:
                    session.vector_store.delete_collection()
                except Exception as e:
                    logger.warning(f"Could not delete collection for session {session_id} on reset: {e}")

            # Clear references
            session.qa_chain = None
            session.current_pdf = None
            session.chat_history = []
            session.docstore = None
            session.retriever = None
            session.vector_store = None
            logger.info(f"RAG pipeline state cleared successfully for session {session_id}.")


# Singleton instance of RAG pipeline
rag_pipeline = RAGPipeline()
