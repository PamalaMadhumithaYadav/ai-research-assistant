from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def inspect_index():
    index_path = "data/faiss_index/"
    
    print(f"Loading FAISS index from {index_path}...")
    # We must initialize the same embedding model used to create the index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        # Load the index
        vector_store = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Access the underlying dictionary of stored documents
        docstore = vector_store.docstore._dict
        
        print("\n" + "="*50)
        print(f"✅ TOTAL CHUNKS IN DATABASE: {len(docstore)}")
        print("="*50 + "\n")
        
        # Let's count how many chunks belong to each paper
        paper_counts = {}
        for doc_id, doc in docstore.items():
            paper_id = doc.metadata.get('paper_id', 'Unknown')
            paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1
            
        print("📊 BREAKDOWN BY PAPER:")
        for paper, count in paper_counts.items():
            print(f" - {paper}: {count} chunks")
            
        print("\n🔍 SAMPLE OF STORED DATA (First 2 chunks):")
        for i, (doc_id, doc) in enumerate(docstore.items()):
            if i >= 2:  # Just look at the first 2 to keep the terminal clean
                break
            print(f"\n--- Chunk {i+1} ---")
            print(f"Internal ID: {doc_id}")
            print(f"Metadata: {doc.metadata}")
            print(f"Text Snippet: {doc.page_content[:150]}...") # Print first 150 chars

    except Exception as e:
        print(f"Error loading index: {e}")

if __name__ == "__main__":
    inspect_index()