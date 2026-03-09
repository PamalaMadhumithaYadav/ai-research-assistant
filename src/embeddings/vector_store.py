import os
import logging
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class FAISSManager:
    """Manages dense vector search and storage for text chunks using FAISS."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initializes the FAISS Embeddings manager."""
        logger.info(f"Initializing HF Embeddings with model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.index_path = "data/faiss_index/"
        self.vector_store: FAISS = None

    def embed_and_store(self, chunks: List[Dict[str, Any]]):
        """
        Embeds texts utilizing LangChain's Document schema into FAISS.
        
        Args:
            chunks: List of dictionaries containing 'text', 'chunk_id', and 'paper_id'.
        """
        if not chunks:
            logger.warning("No chunks provided to FAISS embed_and_store.")
            return

        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "project_id": chunk.get("project_id", "")
                }
            )
            documents.append(doc)

        logger.info(f"Embedding and storing {len(documents)} chunks in FAISS...")
        
        # Append existing store or build the index anew
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
            self.vector_store.add_documents(documents)
        else:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
        logger.debug(f"Successfully embedded {len(documents)} chunks.")

    def save_local(self, directory_path: str):
        """Persists the FAISS index to the specified disk path."""
        if not self.vector_store:
            logger.error("Attempted to save empty vector store.")
            raise ValueError("Vector store is empty. Call embed_and_store first.")
            
        # Ensure the target directory exists
        os.makedirs(directory_path, exist_ok=True)
        
        logger.info(f"Saving vector database locally to: {directory_path}")
        self.vector_store.save_local(directory_path)
