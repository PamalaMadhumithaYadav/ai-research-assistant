import os
import sys
import logging
from dotenv import load_dotenv

# Ensure modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.graph.neo4j_builder import GraphManager
from src.embeddings.vector_store import FAISSManager
from src.retriever.hybrid_search import HybridRetriever

# Configure basic logging formatter
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load local neo4j credentials from .env
load_dotenv()

def run_retrieval_test():
    logger.info("Initializing Retrieval Subsystems...")
    
    # Initialize FAISS and attempt to load local index
    faiss_path = "data/faiss_index/"
    faiss_manager = FAISSManager()
    
    try:
        if os.path.exists(faiss_path):
            logger.info(f"Loading local FAISS index from {faiss_path}")
            # Langchain specific index loading mechanics
            faiss_manager.vector_store = FAISS.load_local("data/faiss_index/", faiss_manager.embeddings, allow_dangerous_deserialization=True)
        else:
            logger.error("FAISS index not found. Please run ingest_pipeline.py first.")
            return
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        return

    # Initialize Neo4j connection
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    graph_manager = GraphManager(uri=neo4j_uri, username=neo4j_user, password=neo4j_password)

    # Boot the orchestrator
    retriever = HybridRetriever(vector_store_manager=faiss_manager, graph_manager=graph_manager)

    try:
        # Define and execute the Test Query
        test_query = "How are Transformers utilized in visual attention?"
        logger.info(f"Executing Test Query:\n >>> {test_query}\n")
        
        # Dispatch Hybrid Context Extraction logic
        final_context = retriever.retrieve_context(query=test_query, top_k=3)
        
        print("\n" + "="*80)
        print(final_context)
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Test Retrieval Pipeline failed: {e}")
    finally:
        logger.info("Closing Graph connection.")
        graph_manager.close()

if __name__ == "__main__":
    run_retrieval_test()
