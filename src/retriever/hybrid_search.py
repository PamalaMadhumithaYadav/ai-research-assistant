import logging
from typing import List, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Orchestrates Hybrid Search combining dense vector semantics with Neo4j graph relationships."""

    def __init__(self, vector_store_manager, graph_manager, rrf_k: int = 60):
        """
        Args:
            vector_store_manager: Initialized FAISSManager instance.
            graph_manager: Initialized GraphManager instance.
            rrf_k: Constant for Reciprocal Rank Fusion.
        """
        self.vector_store_manager = vector_store_manager
        self.graph_manager = graph_manager
        self.rrf_k = rrf_k

    def _semantic_search(self, query: str, top_k: int, project_id: str) -> List[Dict[str, Any]]:
        """Queries the FAISS vector store for semantic matches isolated to a project."""
        logger.debug(f"Executing semantic search for query: '{query}' (top_k={top_k}, project_id={project_id})")
        if not self.vector_store_manager.vector_store:
            logger.warning("Vector store is empty or not initialized properly.")
            return []

        # Langchain similarity search returns standard Document objects
        docs = self.vector_store_manager.vector_store.similarity_search(
            query, 
            k=top_k, 
            filter={"project_id": project_id}
        )
        
        hits = []
        for doc in docs:
            hits.append({
                "chunk_id": doc.metadata.get("chunk_id"),
                "paper_id": doc.metadata.get("paper_id"),
                "text": doc.page_content
            })
        return hits

    def _graph_traversal(self, chunk_ids: List[str], project_id: str) -> List[Dict[str, Any]]:
        """
        Queries Neo4j for entities connected to the papers belonging to the provided chunks.
        Safeguards against graph explosion by limiting collected entities AND forcing project_id scope.
        """
        if not chunk_ids:
            return []

        logger.debug(f"Executing graph traversal for {len(chunk_ids)} chunk(s).")
        
        query = """
        MATCH (c:Chunk)-[:BELONGS_TO]->(p:Paper)
        WHERE c.chunk_id IN $chunk_ids AND c.project_id = $project_id AND p.project_id = $project_id
        MATCH (p)-[r]->(e:Entity)
        WHERE e.project_id = $project_id
        WITH p, type(r) AS rel_type, collect(DISTINCT e.name)[0..10] AS entities
        RETURN p.paper_id AS paper_id, collect({relation: rel_type, entities: entities}) AS graph_context
        """
        
        hits = []
        with self.graph_manager.driver.session() as session:
            result = session.run(query, chunk_ids=chunk_ids, project_id=project_id)
            for record in result:
                hits.append({
                    "paper_id": record["paper_id"],
                    "graph_context": record["graph_context"]
                })
                
        return hits

    def _compute_rrf(self, semantic_hits: List[Dict[str, Any]], graph_hits: List[Dict[str, Any]]) -> List[str]:
        """
        Merges semantic and graph hits using Reciprocal Rank Fusion (RRF).
        Graph ranks are applied at the paper level to their corresponding chunks.
        """
        rrf_scores = defaultdict(float)

        # 1. Distribute Semantic Rank Scores
        # Iterate over enumerate to natively obtain rank placement (0-indexed, so rank+1)
        chunk_to_paper = {}
        for rank, hit in enumerate(semantic_hits):
            chunk_id = hit["chunk_id"]
            paper_id = hit["paper_id"]
            chunk_to_paper[chunk_id] = paper_id
            
            rrf_scores[chunk_id] += 1.0 / (self.rrf_k + (rank + 1))

        # 2. Distribute Graph Rank Scores
        # Graph hits are grouped by paper_id. We apply their rank to all chunk_ids associated with that paper.
        for rank, hit in enumerate(graph_hits):
            paper_id = hit["paper_id"]
            # Find all chunks in the semantic hits that belong to this paper
            related_chunk_ids = [c_id for c_id, p_id in chunk_to_paper.items() if p_id == paper_id]
            
            for chunk_id in related_chunk_ids:
                rrf_scores[chunk_id] += 1.0 / (self.rrf_k + (rank + 1))

        # Sort chunk IDs descending by their computed RRF score
        ranked_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return ranked_chunk_ids

    def retrieve_context(self, query: str, project_id: str, top_k: int = 5) -> str:
        """
        Orchestrates hybrid retrieval and formats an injected prompt context string.
        """
        logger.info(f"Retrieving hybrid context for: '{query}' in project '{project_id}'")
        
        # 1. Semantic Search
        semantic_hits = self._semantic_search(query, top_k, project_id)
        if not semantic_hits:
            return "### RETRIEVED RESEARCH CONTEXT ###\nNo relevant semantic chunks found."

        chunk_ids = [hit["chunk_id"] for hit in semantic_hits]

        # 2. Graph Traversal Context
        graph_hits = self._graph_traversal(chunk_ids, project_id)

        # 3. Reciprocal Rank Fusion
        ranked_chunk_ids = self._compute_rrf(semantic_hits, graph_hits)

        # 4. Map back data and construct string
        # Create quick lookups
        semantic_lookup = {hit["chunk_id"]: hit for hit in semantic_hits}
        graph_lookup = {hit["paper_id"]: hit["graph_context"] for hit in graph_hits}

        context_blocks = []
        for chunk_id in ranked_chunk_ids:
            hit = semantic_lookup[chunk_id]
            paper_id = hit["paper_id"]
            text_content = hit["text"].replace('\n', ' ').strip()
            
            block = f"- [Source: {paper_id} | Chunk: {chunk_id}]\n  Text: {text_content}"
            
            # Inject Graph structural relationships logically
            paper_graph_context = graph_lookup.get(paper_id, [])
            if paper_graph_context:
                graph_descriptions = []
                for context in paper_graph_context:
                    rel = context["relation"]
                    ents = ", ".join(context["entities"])
                    graph_descriptions.append(f"{rel}: [{ents}]")
                block += f"\n  Graph Data: {'; '.join(graph_descriptions)}"
                
            context_blocks.append(block)

        final_context = "### RETRIEVED RESEARCH CONTEXT ###\n" + "\n\n".join(context_blocks)
        logger.info("Successfully assembled Hybrid RRF context.")
        return final_context
