import os
import logging
from typing import Dict, Any

from neo4j import GraphDatabase, Driver

logger = logging.getLogger(__name__)

class GraphManager:
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        """Initializes the Neo4j driver with provided or environment credentials."""
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        
        self.driver: Driver = GraphDatabase.driver(
            self.uri, 
            auth=(self.username, self.password)
        )

    def close(self):
        """Closes the Neo4j driver connection."""
        if self.driver:
            self.driver.close()

    def setup_constraints(self):
        """Executes Cypher to create UNIQUE constraints on Chunk and Paper nodes."""
        constraints_queries = [
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE"
        ]
        with self.driver.session() as session:
            for query in constraints_queries:
                session.run(query)
                logger.info(f"Executed Neo4j constraint: {query.split(' ')[2]}")

    def insert_extraction(self, chunk_id: str, paper_id: str, project_id: str, extraction: Dict[str, Any]):
        """
        Inserts extracted entities and relationships into Neo4j using idempotent MERGE queries.
        Links extracted items to Chunk, and chunks to the corresponding Paper.
        Ensures all nodes are isolated by project_id.
        
        Args:
            chunk_id: The chunk UUID or reference.
            paper_id: The UUID or metadata title of the source paper.
            project_id: The UUID of the workspace isolating the data.
            extraction: Dictionary parsing key_entities, methods_and_frameworks, datasets_and_tools, core_concepts, and relationships.
        """
        base_entities_query = """
        // Merge Paper and Chunk
        MERGE (p:Paper {paper_id: $paper_id, project_id: $project_id})
        MERGE (c:Chunk {chunk_id: $chunk_id, project_id: $project_id})
        MERGE (c)-[:BELONGS_TO]->(p)
        
        // Merge Key Entities
        WITH c
        UNWIND $key_entities AS ent
        MERGE (entity:KeyEntity:Entity {id: toLower(ent), project_id: $project_id})
        ON CREATE SET entity.name = ent
        MERGE (c)-[:MENTIONS]->(entity)
        """

        methods_query = """
        MATCH (c:Chunk {chunk_id: $chunk_id, project_id: $project_id})
        UNWIND $methods_and_frameworks AS meth
        MERGE (method:Method:Entity {id: toLower(meth), project_id: $project_id})
        ON CREATE SET method.name = meth
        MERGE (c)-[:MENTIONS]->(method)
        """

        tools_query = """
        MATCH (c:Chunk {chunk_id: $chunk_id, project_id: $project_id})
        UNWIND $datasets_and_tools AS tool
        MERGE (t:Tool:Entity {id: toLower(tool), project_id: $project_id})
        ON CREATE SET t.name = tool
        MERGE (c)-[:MENTIONS]->(t)
        """

        concepts_query = """
        MATCH (c:Chunk {chunk_id: $chunk_id, project_id: $project_id})
        UNWIND $core_concepts AS concept
        MERGE (co:Concept:Entity {id: toLower(concept), project_id: $project_id})
        ON CREATE SET co.name = concept
        MERGE (c)-[:MENTIONS]->(co)
        """

        # Custom rels: Link from Paper directly if source = 'THIS_PAPER'
        paper_rels_query = """
        MATCH (p:Paper {paper_id: $paper_id, project_id: $project_id})
        MATCH (c:Chunk {chunk_id: $chunk_id, project_id: $project_id})
        UNWIND $rels AS rel

        MERGE (target:Entity {id: toLower(rel.target_entity), project_id: $project_id})
        ON CREATE SET target.name = rel.target_entity
        MERGE (c)-[:MENTIONS]->(target)

        WITH p, target, rel
        CALL apoc.merge.relationship(p, toUpper(replace(rel.relationship_type, ' ', '_')), {}, {}, target, {}) YIELD rel AS r
        RETURN count(r)
        """

        # Custom rels: Standard entity to entity pairs
        entity_rels_query = """
        MATCH (c:Chunk {chunk_id: $chunk_id, project_id: $project_id})
        UNWIND $rels AS rel

        MERGE (source:Entity {id: toLower(rel.source_entity), project_id: $project_id})
        ON CREATE SET source.name = rel.source_entity
        MERGE (c)-[:MENTIONS]->(source)

        MERGE (target:Entity {id: toLower(rel.target_entity), project_id: $project_id})
        ON CREATE SET target.name = rel.target_entity
        MERGE (c)-[:MENTIONS]->(target)

        WITH source, target, rel
        CALL apoc.merge.relationship(source, toUpper(replace(rel.relationship_type, ' ', '_')), {}, {}, target, {}) YIELD rel AS r
        RETURN count(r)
        """

        paper_rels = []
        entity_rels = []
        for rel in extraction.get("relationships", []):
            if rel.get("source_entity") == "THIS_PAPER":
                paper_rels.append(rel)
            else:
                entity_rels.append(rel)

        with self.driver.session() as session:
            # 1. Guarantee Base Merge (Paper + Chunk) and conditionally Key Entities
            session.run(base_entities_query, 
                        paper_id=paper_id, 
                        chunk_id=chunk_id, 
                        project_id=project_id,
                        key_entities=extraction.get("key_entities", []))
            
            # 2. Insert Methods and Frameworks
            if extraction.get("methods_and_frameworks"):
                session.run(methods_query, chunk_id=chunk_id, project_id=project_id, methods_and_frameworks=extraction.get("methods_and_frameworks"))
            
            # 3. Insert Datasets and Tools
            if extraction.get("datasets_and_tools"):
                session.run(tools_query, chunk_id=chunk_id, project_id=project_id, datasets_and_tools=extraction.get("datasets_and_tools"))

            # 4. Insert Core Concepts
            if extraction.get("core_concepts"):
                session.run(concepts_query, chunk_id=chunk_id, project_id=project_id, core_concepts=extraction.get("core_concepts"))
            
            # 4. Insert dynamic Paper relationships
            if paper_rels:
                session.run(paper_rels_query, chunk_id=chunk_id, paper_id=paper_id, project_id=project_id, rels=paper_rels)

            # 5. Insert dynamic Entity relationships
            if entity_rels:
                session.run(entity_rels_query, chunk_id=chunk_id, project_id=project_id, rels=entity_rels)
            
            logger.debug(f"Committed graph inserts for chunk -> {chunk_id}")
