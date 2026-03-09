import os
import sys
import logging
from dotenv import load_dotenv

# Ensure modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.graph.neo4j_builder import GraphManager
from src.ingestion.parser import PDFProcessor
from src.embeddings.vector_store import FAISSManager
from src.graph.extractor import extract_entities_from_text

# Configure basic logging formatter
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load local neo4j credentials from .env
load_dotenv()

def process_documents(uploaded_files, project_id: str):
    # 1. Pipeline Orchestrators initialization
    logger.info("Initializing Data & Graph Pipeline Services...")
    
    pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
    faiss_manager = FAISSManager()
    
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    graph_manager = GraphManager(uri=neo4j_uri, username=neo4j_user, password=neo4j_password)
    
    faiss_path = "data/faiss_index/"
    
    try:
        # Pre-execution DB Constraint verifications
        logger.info("Setting up database constraints...")
        graph_manager.setup_constraints()
        
        all_chunks = []
        # Save Streamlit files momentarily
        os.makedirs("data/raw_pdfs", exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data/raw_pdfs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # 3. Read and format PDF File into Chunk documents
            logger.info(f"Starting pipeline on PDF '{file_path}'")
            chunks = pdf_processor.extract_and_chunk(file_path, project_id)
            all_chunks.extend(chunks)
            logger.info(f"Extracted {len(chunks)} text chunks.")
            
        if not all_chunks:
            logger.warning("No chunks were extracted.")
            return

        # 4. Dense Vector store Embedding and persistence
        faiss_manager.embed_and_store(all_chunks)
        faiss_manager.save_local(faiss_path)
        
        # 5. Iterative Graph relationships formulation
        logger.info("Commencing LLM Structure extraction and Graph indexing...")
        for chunk in all_chunks:
            chunk_id = chunk["chunk_id"]
            paper_id = chunk["paper_id"]
            text = chunk["text"]
            
            logger.debug(f"Extracting relationships for chunk {chunk_id}...")
            # Simulate real LLM API call using predefined Pydantic schemas
            extraction = extract_entities_from_text(text)
            
            # Commit logic to Graph store
            graph_manager.insert_extraction(
                chunk_id=chunk_id, 
                paper_id=paper_id, 
                project_id=project_id,
                extraction=extraction
            )
            
        logger.info("Pipeline execution mapped and completely successfully.")

    except Exception as e:
        logger.error(f"Ingestion Pipeline unhandled fault: {e}")
    finally:
        logger.info("Closing Neo4j GraphManager connection...")
        graph_manager.close()
