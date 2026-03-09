import os
import uuid
import logging
import fitz  # PyMuPDF
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles extracting text from PDFs and chunking them for vector/graph processing."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_and_chunk(self, pdf_path: str, project_id: str) -> List[Dict[str, Any]]:
        """
        Reads a PDF, extracts all text, splits it into overlapping chunks.
        
        Args:
            pdf_path: Absolute or relative path to the PDF file.
            project_id: The unique identifier for the workspace.
            
        Returns:
            A list of dictionary chunks with 'chunk_id', 'paper_id', 'project_id', and 'text'.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

        paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
        logger.info(f"Extracting text from PDF: {pdf_path} (paper_id: {paper_id})")

        full_text = []
        try:
            with fitz.open(pdf_path) as pdf:
                for page_num in range(len(pdf)):
                    page = pdf.load_page(page_num)
                    text = page.get_text("text")
                    if text:
                        full_text.append(text)
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise

        combined_text = "\n".join(full_text)
        logger.debug(f"Total extracted text length: {len(combined_text)} characters")

        # Split text into overlapping LangChain semantic chunks
        split_texts = self.text_splitter.split_text(combined_text)
        logger.info(f"Split document into {len(split_texts)} chunks")

        # Format into the expected dictionary structure
        chunks = []
        for text_segment in split_texts:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "paper_id": paper_id,
                "project_id": project_id,
                "text": text_segment
            })

        return chunks
