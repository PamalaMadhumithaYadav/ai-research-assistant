import logging
from typing import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Use neo4j_builder instead of database as it is the real file name
from src.embeddings.vector_store import FAISSManager
from src.graph.neo4j_builder import GraphManager
from src.retriever.hybrid_search import HybridRetriever
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

# ==========================================
# 1. State and Schemas
# ==========================================
class ResearchState(TypedDict):
    """LangGraph State passed between research nodes."""
    user_query: str
    search_plan: str
    retrieved_context: str
    draft_answer: str
    project_id: str
    is_conversational: bool
    critique: str
    is_faithful: bool
    revision_count: int

class CriticOutput(BaseModel):
    """Pydantic V2 schema for structured critique output."""
    is_faithful: bool = Field(description="True if the draft is fully supported by the retrieved_context, False otherwise.")
    critique: str = Field(description="Detailed feedback on what is missing or hallucinated, or 'Looks good' if faithful.")

from dotenv import load_dotenv

# Initialize the Gemini language model (requires GOOGLE_API_KEY environment variable)
load_dotenv()
base_url = "https://presentation-progressive-marketplace-experiencing.trycloudflare.com/"

# 1. The Manager: General reasoning for Planning and Critiquing
orchestrator_llm = ChatOllama(model="mistral", temperature=0, base_url=base_url)

# 2. The Specialist: Strict citation enforcement for Synthesizing
specialist_llm = ChatOllama(model="my-research-agent", temperature=0, base_url=base_url)

# ==========================================
# 2. Node Implementations
# ==========================================
class RouterOutput(BaseModel):
    is_conversational: bool = Field(description="True if the query is conversational/chit-chat, False if it requires scientific research.")

def router_node(state: ResearchState) -> dict:
    """Classifies if the user query is conversational or requires research."""
    logger.info("--- NODE: ROUTER ---")
    query = state["user_query"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent router. Determine if the following user query is standard conversational chit-chat (like greetings or polite remarks) or an actual research question requiring literature retrieval."),
        ("user", "Query: {query}")
    ])
    
    router = prompt | orchestrator_llm.with_structured_output(RouterOutput)
    result: RouterOutput = router.invoke({"query": query})
    logger.debug(f"Router output -> Conversational: {result.is_conversational}")
    
    return {"is_conversational": result.is_conversational}

def conversational_node(state: ResearchState) -> dict:
    """Handles simple conversational greetings natively."""
    logger.info("--- NODE: CONVERSATIONAL ---")
    query = state["user_query"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and polite scientific AI research assistant. Respond warmly to the user's greeting or polite interaction."),
        ("user", "{query}")
    ])
    
    response = (prompt | orchestrator_llm).invoke({"query": query})
    answer = response.content if isinstance(response.content, str) else response.content[0].get("text", str(response.content))
    return {"draft_answer": answer, "is_conversational": True}

def planner_node(state: ResearchState) -> dict:
    """Extracts key concepts and outputs a search plan."""
    logger.info("--- NODE: PLANNER ---")
    query = state["user_query"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research planner. Extract the core entities and relationships from the user query to form a concise search plan."),
        ("user", "Query: {query}")
    ])
    
    chain = prompt | orchestrator_llm
    response = chain.invoke({"query": query})
    
    plan_text = response.content if isinstance(response.content, str) else response.content[0].get("text", str(response.content))
    
    logger.debug(f"Generated Search Plan: {plan_text}")
    return {"search_plan": plan_text}

def retriever_node(state: ResearchState) -> dict:
    """
    Retrieves semantic and graph context using the real HybridRetriever.
    """
    logger.info("--- NODE: RETRIEVER ---")
    search_plan = state.get("search_plan", "")
    project_id = state.get("project_id", "")
    
    import os
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize the manager
    faiss_manager = FAISSManager()
    
    # Load the FAISS index safely
    faiss_manager.vector_store = FAISS.load_local("data/faiss_index/", embeddings, allow_dangerous_deserialization=True)
    
    # Initialize Neo4j GraphManager
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    graph_manager = GraphManager(uri=neo4j_uri, username=neo4j_user, password=neo4j_password)
    
    # Initialize Hybrid Retriever using the manager
    hybrid_retriever = HybridRetriever(faiss_manager, graph_manager)
    
    try:
        # Actually retrieve the context from the dual systems
        retrieved_context = hybrid_retriever.retrieve_context(search_plan, project_id)
        logger.debug("Successfully retrieved context from Hybrid Search.")
    except Exception as e:
        logger.error(f"Hybrid retrieval failed: {e}")
        retrieved_context = "Error during retrieval."
    finally:
        graph_manager.close()
        
    return {"retrieved_context": retrieved_context}

def synthesizer_node(state: dict) -> dict: 
    logger.info("--- NODE: SYNTHESIZER ---")
    
    query = state["user_query"] 
    context = state.get("retrieved_context", "")
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    # =================================================================
    # STEP 1: THE SPECIALIST BRAIN (Extraction & Citation Enforcement)
    # =================================================================
    logger.info("[INFO] Specialist Brain: Extracting raw facts and citations...")
    
    specialist_system_prompt = (
        "You are a strict scientific data extractor. Extract the entities from the Context that relate to the Question. "
        "You MUST append the citation '[Source: X]' at the end of your answer."
    )
    
    if critique:
        specialist_system_prompt += f"\n\nCRITIQUE RECEIVED ON PREVIOUS DRAFT: {critique}\nPlease adjust your extraction."

    specialist_prompt = ChatPromptTemplate.from_messages([
        ("system", specialist_system_prompt),
        ("user", "Context: {context}\n\nQuery: {query}")
    ])
    
    specialist_chain = specialist_prompt | specialist_llm
    specialist_response = specialist_chain.invoke({"context": context, "query": query})
    
    raw_draft = specialist_response.content if isinstance(specialist_response.content, str) else specialist_response.content[0].get("text", str(specialist_response.content))

    # =================================================================
    # STEP 2: THE MANAGER BRAIN (Chain-of-Thought Verification)
    # =================================================================
    logger.info("[INFO] Manager Brain: Verifying context and polishing response...")
    
    manager_system_prompt = (
        "You are a strict AI scientific editor. I will provide a User Question, the Original Context, and a Raw Draft extracted by an automated system.\n\n"
        "CRITICAL RULES:\n"
        "1. You MUST first think step-by-step inside <thinking> tags. Compare the specific subject of the User Question to the Original Context. Does the Context actually mention the subject?\n"
        "2. If the Context DOES NOT mention the subject of the User Question, your final output after the </thinking> tag MUST be exactly: 'I cannot answer this based on the retrieved context.'\n"
        "3. If the Context DOES contain the answer, rewrite the Raw Draft into a clear paragraph after the </thinking> tag, keeping the [Source: X] citation."
    )

    manager_prompt = ChatPromptTemplate.from_messages([
        ("system", manager_system_prompt),
        ("user", "User Question: {query}\n\nOriginal Context:\n{context}\n\nRaw Draft:\n{raw_draft}")
    ])

    manager_chain = manager_prompt | orchestrator_llm
    manager_response = manager_chain.invoke({"query": query, "context": context, "raw_draft": raw_draft})

    # Extract the string content
    raw_manager_text = manager_response.content if isinstance(manager_response.content, str) else manager_response.content[0].get("text", str(manager_response.content))

    # THE PYTHON FIX: Strip out the <thinking> tags so the user only sees the final answer
    if "</thinking>" in raw_manager_text:
        final_draft_text = raw_manager_text.split("</thinking>")[-1].strip()
    else:
        final_draft_text = raw_manager_text.strip()

    logger.debug(f"Generated Draft Answer (Revision {revision_count + 1})")
    return {"draft_answer": final_draft_text, "revision_count": revision_count + 1}

def critic_node(state: ResearchState) -> dict:
    """Evaluates the draft answer against the retrieved context."""
    logger.info("--- NODE: CRITIC ---")
    draft = state.get("draft_answer", "")
    context = state.get("retrieved_context", "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an evaluator. Determine whether the drafted answer is strictly faithful to the provided context. Do not evaluate whether the answer is true in the real world, only if it is supported by the context."),
        ("user", "Context: {context}\n\nDraft Answer: {draft}")
    ])
    
    # Force structured Pydantic output
    evaluator = prompt | orchestrator_llm.with_structured_output(CriticOutput)
    eval_result: CriticOutput = evaluator.invoke({"context": context, "draft": draft})
    
    logger.debug(f"Critique Output -> Faithful: {eval_result.is_faithful} | Critique: {eval_result.critique}")
    return {"is_faithful": eval_result.is_faithful, "critique": eval_result.critique}

# ==========================================
# 3. Conditional Routing
# ==========================================
def route_after_router(state: ResearchState) -> str:
    """Routes to conversational node or planner node."""
    if state.get("is_conversational"):
        logger.info("--- ROUTE: CONVERSATIONAL. BYPASSING PIPELINE. ---")
        return "conversational_node"
    logger.info("--- ROUTE: RESEARCH QUERY. PROCEEDING TO PLANNER. ---")
    return "planner_node"

def route_after_critic(state: ResearchState) -> str:
    """Determines whether to finish or iterate based on the critic's output."""
    if state.get("is_faithful") is True:
        logger.info("--- ROUTE: CRITIC APPROVED. FINISHING. ---")
        return END
    
    rev_count = state.get("revision_count", 0)
    if rev_count >= 3:
        logger.warning(f"--- ROUTE: MAX REVISIONS ({rev_count}) REACHED. FINISHING. ---")
        return END
    
    logger.info("--- ROUTE: CRITIC REJECTED. RETURNING TO SYNTHESIZER. ---")
    return "synthesizer_node"

# ==========================================
# 4. Graph Construction
# ==========================================
def build_research_graph():
    """Builds and compiles the LangGraph StateGraph."""
    graph = StateGraph(ResearchState)
    
    # Add Nodes
    graph.add_node("router_node", router_node)
    graph.add_node("conversational_node", conversational_node)
    graph.add_node("planner_node", planner_node)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("synthesizer_node", synthesizer_node)
    graph.add_node("critic_node", critic_node)
    
    # Add Edges (Linear Flow)
    graph.set_entry_point("router_node")
    graph.add_conditional_edges("router_node", route_after_router)
    
    graph.add_edge("conversational_node", END)
    
    graph.add_edge("planner_node", "retriever_node")
    graph.add_edge("retriever_node", "synthesizer_node")
    graph.add_edge("synthesizer_node", "critic_node")
    
    # Add Conditional Edge (Cyclical critique loop)
    graph.add_conditional_edges("critic_node", route_after_critic)
    
    return graph.compile()
