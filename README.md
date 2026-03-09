# 🧠 Autonomous AI Research Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-blue?style=for-the-badge&logo=neo4j)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)

## 📌 Project Overview
A multi-tenant GraphRAG (Retrieval-Augmented Generation) SaaS application that allows users to upload scientific research papers (PDFs), dynamically extract domain-agnostic knowledge graphs, and query them using an autonomous LangGraph agent. Built with strict hallucination-prevention mechanisms, it empowers researchers to synthesize cited insights across disparate documents within secure, isolated workspaces.

## 🏗️ System Architecture

*   **Frontend**: Built with **Streamlit** to provide a multi-project dashboard, seamless document upload, and real-time visualization of the AI agent's internal reasoning process.
*   **Backend Orchestrator**: Powered by **LangGraph**, utilizing a stateful cyclical agent flow:
    *   **Conversational Router**: Intelligently bypasses heavy research logic for standard greetings/chit-chat.
    *   **Planner Node**: Formulates a structured search trajectory based on the user's core query.
    *   **Retriever Node**: Executes Hybrid Search algorithms.
    *   **Synthesizer Node**: Uses a "Specialist" LLM to extract rigorous cited facts, and a "Manager" LLM to weave them into a polished response using Chain-of-Thought.
    *   **Critic Node**: A strict verification gatekeeper that evaluates the draft against the retrieved context to entirely prevent hallucinations.
*   **Knowledge Graph (Cloud)**: Backed by **Neo4j AuraDB**, mapping theoretical concepts, datasets, models, and methods natively into interconnected nodes.
*   **Vector Store (Local)**: **FAISS** index for dense semantic similarity search.
*   **Retrieval Strategy**: Hybrid Search combining FAISS vector embeddings and Neo4j Cypher graph traversals, seamlessly merged via Reciprocal Rank Fusion (RRF).
*   **LLM Engine & Infrastructure**: Runs **Local Mistral 7B** served via **Ollama** on a Google Colab GPU (e.g., A100) and tunneled locally via **Cloudflare Tunnels**. Implements a robust "Two-Brain" architecture (Specialist for extraction + Manager for conversational verification).

## 🛠️ Model Fine-Tuning & Hosting
This project leverages **Mistral-7B**, which was explicitly fine-tuned on GraphRAG-specific extraction samples using QLoRA. 
The repository (and your workflow) utilizes dedicated Google Colab notebooks for:
*   **Fine-Tuning**: Exporting targeted graph instruction data and training the `my-research-agent` adapter.
*   **Hosting**: Exposing the local GPU inference natively to the local Streamlit frontend via a secure, encrypted **Cloudflare Tunnel**.
This allows for incredibly fast, API-cost-free, secure document extraction right from your local machine.

## ✨ Key Features
*   **Dynamic Domain-Agnostic Extraction**: Parses structural schemas dynamically (e.g., `key_entities`, `methods_and_frameworks`, `datasets_and_tools`, `core_concepts`), allowing ingestion of Biology, Physics, or Finance literature without bias.
*   **Zero-Hallucination Critic Loop**: The cyclical LangGraph architecture actively rejects and rewrites unfaithful drafts before the user ever sees them.
*   **Strict Multi-Tenancy**: Guaranteed graph and vector isolation using UUID `project_id` tagging, enabling simultaneous decoupled workspaces.

## 📋 Prerequisites
Before you begin, ensure you have the following requirements:
*   **Python 3.10+** installed on your local machine.
*   A **Neo4j AuraDB** cloud account (Free Tier works perfectly) to host the Knowledge Graph.
*   A **Google Colab** account (to run the Mistral model on a free T4/A100 GPU for offline fine-tuning and inference).

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ai-research-assistant.git
   cd ai-research-assistant
   ```

2. **Set up the virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory. Use the template below:
   ```env
   # .env template
   
   # Neo4j AuraDB Credentials
   NEO4J_URI=neo4j+s://<YOUR_DB_ID>.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_secure_password
   
   # LLM Endpoint (From your Cloudflare Tunnel)
   OLLAMA_BASE_URL=https://your-tunnel-name.trycloudflare.com/
   ```

## 💻 Usage

1. **Launch the LLM Backend (Google Colab):**
   *   Open your hosting Colab Notebook.
   *   Execute the cells to deploy Ollama and load the `mistral` and fine-tuned `my-research-agent` models.
   *   Start the Cloudflare tunnel in the notebook.
   *   Copy the generated `trycloudflare.com` URL.

2. **Connect the Application:**
   *   Open your local `.env` file.
   *   Update `OLLAMA_BASE_URL` with the URL generated by Cloudflare.
   
   *(Note: The Orchestrator and Extractor read directly from standard variables configured for `OLLAMA_BASE_URL` or fallback gracefully based on the repository implementation).*

3. **Start the Application:**
   ```bash
   streamlit run app.py
   ```
   *   Create a workspace in the sidebar.
   *   Upload your PDFs and click **Ingest Documents**.
   *   Start querying your fully autonomous, private Research Graph!
