import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class Relationship(BaseModel):
    """Schema representing a logical relationship in the text."""
    source_entity: str = Field(..., description="Entity initiating the relation. E.g. 'THIS_PAPER', 'Algorithm', 'Protein'")
    target_entity: str = Field(..., description="Target entity of the relation. E.g. 'Economic Model', 'Dataset'")
    relationship_type: str = Field(..., description="Type of interaction. E.g. 'USES_METHOD', 'STUDIES_PHENOMENON'")

class ChunkExtraction(BaseModel):
    """Schema representing the generalized knowledge extracted from a single text chunk."""
    key_entities: List[str] = Field(default_factory=list, description="Primary subjects or core entities discussed. E.g. 'Protein', 'Company', 'Algorithm'.")
    methods_and_frameworks: List[str] = Field(default_factory=list, description="Algorithmic, mathematical, or scientific methods/protocols mentioned.")
    datasets_and_tools: List[str] = Field(default_factory=list, description="Specific datasets, physical tools, software, or clinical instruments utilized.")
    core_concepts: List[str] = Field(default_factory=list, description="General theoretical or scientific concepts. E.g. 'Neuroplasticity', 'Inflation', 'Quantum Entanglement'.")
    relationships: List[Relationship] = Field(default_factory=list, description="List of directed logical relationships between entities.")

def extract_entities_from_text(text: str) -> dict:
    """
    Uses ChatOllama to extract a Pydantic-validated dict matching `ChunkExtraction`.
    
    Args:
        text: The text chunk to extract entities from.
        
    Returns:
        dict: A dictionary structure adhering to the ChunkExtraction model.
    """
    logger.debug(f"Extracting from text of length {len(text)} using LLM...")
    
    # We use the same local Mistral setup as the orchestrator
    base_url = "https://apart-aside-starter-rivers.trycloudflare.com/"
    llm = ChatOllama(model="mistral", temperature=0, base_url=base_url)
    
    system_prompt = (
        "You are an expert scientific data extractor. Your task is to extract "
        "structured entities and relationships from the provided research text. "
        "Return the output perfectly conforming to the strict JSON schema provided."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Text to extract from:\n\n{text}")
    ])
    
    # Force structured output using the Pydantic schema
    extractor_chain = prompt | llm.with_structured_output(ChunkExtraction)
    
    try:
        extraction: ChunkExtraction = extractor_chain.invoke({"text": text})
        return extraction.model_dump()
    except Exception as e:
        logger.error(f"Failed to extract structured entities: {e}")
        # Fallback to empty structure to prevent pipeline crash
        return ChunkExtraction().model_dump()
