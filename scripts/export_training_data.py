import os
import sys
import json
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

# Ensure modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.graph.neo4j_builder import GraphManager
from src.embeddings.vector_store import FAISSManager

# Configure basic logging formatter
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load local credentials from .env
load_dotenv()

def export_training_data():
    """Extracts hybrid FAISS+Neo4j data to generate Mistral instruction-tuning JSONL."""
    
    output_file = "data/mistral_training_data.jsonl"
    logger.info("Initializing components for Training Data Export...")
    
    # 1. Initialize FAISS Manager & Vector Store
    faiss_path = "data/faiss_index/"
    faiss_manager = FAISSManager()
    
    if os.path.exists(faiss_path):
        logger.info(f"Loading local FAISS index from {faiss_path}")
        faiss_manager.vector_store = FAISS.load_local(faiss_path, faiss_manager.embeddings, allow_dangerous_deserialization=True)
    else:
        logger.error("FAISS index not found. Please run ingest_pipeline.py first.")
        return

    # 2. Initialize Neo4j GraphManager
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    graph_manager = GraphManager(uri=neo4j_uri, username=neo4j_user, password=neo4j_password)

    # 3. Fetch all chunks from FAISS
    # Hack: To get all docs from a FAISS index in Langchain without a search query, 
    # we can directly access the underlying docstore dictionary.
    docstore_dict = faiss_manager.vector_store.docstore._dict
    all_documents = list(docstore_dict.values())
    
    logger.info(f"Found {len(all_documents)} chunks in FAISS index.")
    
    training_samples = []
    
    try:
        # 4. Iterate and query Graph for each Chunk
        for doc in all_documents:
            chunk_id = doc.metadata.get("chunk_id")
            paper_id = doc.metadata.get("paper_id")
            chunk_text = doc.page_content.replace('\n', ' ').strip()
            
            # Neo4j Query: Get entities linked to the paper this chunk belongs to
            query = """
            MATCH (c:Chunk)-[:BELONGS_TO]->(p:Paper)
            WHERE c.chunk_id = $chunk_id
            MATCH (p)-[r]->(e:Entity)
            WITH p, type(r) AS rel_type, collect(DISTINCT e.name)[0..10] AS entities
            RETURN p.paper_id AS paper_id, collect({relation: rel_type, entities: entities}) AS graph_context
            """
            
            graph_entities_str = ""
            
            with graph_manager.driver.session() as session:
                result = session.run(query, chunk_id=chunk_id)
                for record in result:
                    graph_context = record["graph_context"]
                    
                    graph_descriptions = []
                    for context in graph_context:
                        rel = context["relation"]
                        ents = ", ".join(context["entities"])
                        graph_descriptions.append(f"{rel}: [{ents}]")
                    
                    if graph_descriptions:
                        graph_entities_str = "; ".join(graph_descriptions)

            # 5. Format to Mistral Instruction Prompt
            # "<s>[INST] You are a strict scientific assistant. Using ONLY the Context, answer the Question.\nContext: {chunk_text}\nGraph Data: {graph_entities}\nQuestion: What are the key entities and methods discussed in this context? [/INST] Based on the context, the key entities are {graph_entities}. The text notes: {chunk_text_summary} [Source: {paper_id}].</s>"
            
            # If graph_entities_str is empty, provide a fallback.
            entities_display = graph_entities_str if graph_entities_str else "None detected"
            
            # For the target generation target {chunk_text_summary}, we'll output a truncated portion of the text directly
            # since we don't have a live summarization agent running during this export script.
            chunk_summary_stub = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            
            mistral_prompt = (
                f"<s>[INST] You are a strict scientific assistant. Using ONLY the Context, answer the Question.\n"
                f"Context: {chunk_text}\n"
                f"Graph Data: {entities_display}\n"
                f"Question: What are the key entities and methods discussed in this context? [/INST] "
                f"Based on the context, the key entities are {entities_display}. The text notes: {chunk_summary_stub} [Source: {paper_id}].</s>"
            )
            
            training_samples.append({"text": mistral_prompt})
            
        # 6. Save JSONL File
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in training_samples:
                f.write(json.dumps(sample) + "\n")
                
        logger.info(f"Successfully exported {len(training_samples)} training samples to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during data export pipeline: {e}")
    finally:
        logger.info("Closing Graph connection.")
        graph_manager.close()

if __name__ == "__main__":
    export_training_data()
