import streamlit as st
import logging
import uuid
import time
import os
import sys

# Ensure imports work from src correctly
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from src.agents.orchestrator import app as agent_app
except ImportError:
    from src.agents.orchestrator import build_research_graph
    agent_app = build_research_graph()

from scripts.ingest_pipeline import process_documents

# Configure basic logging formatter
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="AI Research Assistant", page_icon="🧠", layout="wide")

# ==========================================
# 1. State Management
# ==========================================
if "projects" not in st.session_state:
    # project_id -> {"name": str, "messages": list, "files": list}
    st.session_state.projects = {}

if "active_project_id" not in st.session_state:
    st.session_state.active_project_id = None

def create_new_project(name: str):
    """Creates a new project and sets it as active."""
    project_id = str(uuid.uuid4())
    st.session_state.projects[project_id] = {
        "name": name,
        "messages": [],
        "files": []
    }
    st.session_state.active_project_id = project_id
    st.rerun()

# ==========================================
# 2. Sidebar Structure
# ==========================================
with st.sidebar:
    st.title("🧠 Workspaces")
    
    if st.session_state.projects:
        project_names = [data["name"] for pid, data in st.session_state.projects.items()]
        project_ids = list(st.session_state.projects.keys())
        
        # Get the index of the currently active project
        current_index = 0
        if st.session_state.active_project_id in project_ids:
            current_index = project_ids.index(st.session_state.active_project_id)
            
        selected_name = st.selectbox("Select Project", options=project_names, index=current_index)
        
        # Update active project if selection changed
        for pid, pdata in st.session_state.projects.items():
            if pdata["name"] == selected_name:
                if st.session_state.active_project_id != pid:
                    st.session_state.active_project_id = pid
                    st.rerun()
                break
    else:
        st.write("No projects yet.")
        
    st.divider()

    # Active Project Management
    active_id = st.session_state.active_project_id
    if active_id:
        active_project = st.session_state.projects[active_id]
        st.subheader(f"📂 {active_project['name']}")
        
        # File Uploading
        uploaded_files = st.file_uploader("Add Documents (+)", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Ingest Documents"):
                with st.spinner("Processing documents into Knowledge Graph... This may take a moment."):
                    try:
                        # Process files synchronously using real dual-store indexer
                        process_documents(uploaded_files, active_id)
                        
                        for file in uploaded_files:
                            if file.name not in active_project["files"]:
                                active_project["files"].append(file.name)
                                
                        st.success(f"Successfully ingested {len(uploaded_files)} document(s)!")
                        time.sleep(1.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
                    
        # List ingested files
        st.markdown("**Ingested Knowledge Base:**")
        if active_project["files"]:
            for fname in active_project["files"]:
                st.markdown(f"- 📄 `{fname}`")
        else:
            st.caption("No documents ingested yet.")

# ==========================================
# 3. Main Interface Logic
# ==========================================
if not st.session_state.active_project_id:
    # -----------------------------------
    # Landing Page
    # -----------------------------------
    st.title("Welcome to the Autonomous AI Research Assistant")
    st.markdown("Create a dedicated workspace to ingest literature, map entities into a dynamic Knowledge Graph, and converse with your documents using our specialized Local Reasoning Engine.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("### Create a New Project")
        with st.form("new_project_form"):
            new_project_name = st.text_input("Project Name", placeholder="e.g., Transformer Architectures Study")
            submit_button = st.form_submit_button("Create Workspace")
            
            if submit_button:
                if new_project_name.strip():
                    create_new_project(new_project_name.strip())
                else:
                    st.error("Please enter a valid project name.")

else:
    # -----------------------------------
    # Active Chat Interface
    # -----------------------------------
    active_id = st.session_state.active_project_id
    project_data = st.session_state.projects[active_id]
    
    st.title(project_data["name"])
    st.caption("Chat with your documents via the Dual-Brain LangGraph API.")
    
    # Display chat messages tied to specific project
    for message in project_data["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input(f"Ask a question to query {project_data['name']}..."):
        # Add user message to UI
        st.chat_message("user").markdown(prompt)
        
        # Add user message to history
        project_data["messages"].append({"role": "user", "content": prompt})

        # Process assistant response
        with st.chat_message("assistant"):
            expander = st.expander("🧠 Agent Thought Process", expanded=True)
            
            state = {
                "user_query": prompt,
                "project_id": active_id,
                "search_plan": "",
                "retrieved_context": "",
                "draft_answer": "",
                "critique": "",
                "is_faithful": False,
                "revision_count": 0
            }
            
            final_answer = ""
            
            with expander:
                st.write("**Initiating Orchestrator...**")
                
                try:
                    for output in agent_app.stream(state):
                        for node_name, node_state in output.items():
                            st.markdown(f"### ⚙️ Executing Node: `{node_name}`")
                            
                            if "draft_answer" in node_state:
                                final_answer = node_state["draft_answer"]
                                st.write(f"*Draft Generated (Revision: {node_state.get('revision_count', '?')})*")
                                
                            if "search_plan" in node_state:
                                st.markdown(f"**Search Plan:** {node_state['search_plan']}")
                                
                            if "retrieved_context" in node_state:
                                context = node_state['retrieved_context']
                                sneak_peak = context[:200] + "..." if len(context) > 200 else context
                                st.write(f"**Retrieved Context:** {sneak_peak}")
                                
                            if "critique" in node_state:
                                is_faithful = node_state.get('is_faithful', False)
                                status_icon = "✅" if is_faithful else "❌"
                                st.markdown(f"**Critic Feedback:** {status_icon} {node_state['critique']} *(Faithful: {is_faithful})*")
                    
                    st.write("✅ **Execution complete.**")
                    
                except Exception as e:
                    st.error(f"Error during agent execution: {e}")
                    logging.error(f"Agent execution failed: {e}")
                    final_answer = "Sorry, I encountered an error while processing your request."
                
            if final_answer:
                st.markdown(final_answer)
                project_data["messages"].append({"role": "assistant", "content": final_answer})
