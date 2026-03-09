import os
import sys
import logging
from dotenv import load_dotenv

# Ensure modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.agents.orchestrator import build_research_graph

# Configure basic logging formatter
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load local credentials from .env
load_dotenv()

def run_agent_test():
    """Executes a trace test of the LangGraph Orchestrator."""
    
    # Ensure GOOGLE_API_KEY is available during runtime
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY is missing from environment. Exiting test.")
        return

    logger.info("Compiling Research Agent Graph...")
    agent_app = build_research_graph()
    
    # Define Initial State
    initial_state = {
        "user_query": "How are Transformers utilized in protein localization?",
        "revision_count": 0
    }
    
    logger.info(f"Dispatching Test Query:\n >>> '{initial_state['user_query']}'\n")
    print("="*80)
    
    # Stream events through the compiled LangGraph execution cycle
    for state_update in agent_app.stream(initial_state):
        # We grab the node name and its corresponding payload cleanly
        for node_name, state_payload in state_update.items():
            print(f"\n[EXECUTED NODE: {node_name}]")
            
            # Print only exactly what the node updated in the state dict
            for key, value in state_payload.items():
                if key == "draft_answer":
                    print(f" >> {key}:\n{value}")
                else:
                    print(f" >> {key}: {value}")
                
    print("\n" + "="*80)
    logger.info("Agent test execution cycle wrapped.")

if __name__ == "__main__":
    run_agent_test()
