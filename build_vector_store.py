import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import os

def build_store():
    # 1. Initialize Sentence Transformer model
    print("Loading embedding model: all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 2. Initialize ChromaDB persistent client
    db_path = './chroma_db'
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    
    print(f"Initializing ChromaDB client at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    
    # Create or get the collection
    collection = client.get_or_create_collection(name="finqa")

    # 3. Load the FinQA dataset (subset)
    print("Fetching FinQA dataset from Hugging Face...")
    # Using the same version-safe loading as download_data.py
    dataset = load_dataset("ibm/finqa", split="train[:100]", trust_remote_code=True)

    # 4. Loop through examples and add to ChromaDB
    print(f"Encoding and indexing {len(dataset)} documents...")
    
    for i, example in enumerate(dataset):
        # Join the 'pre_text' list into a single string for embedding
        content = " ".join(example['pre_text'])
        
        # Generate embedding
        embedding = model.encode(content).tolist()
        
        # Add to collection
        collection.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "finqa_id": example['id'],
                "answer": example['answer'],
                "question": example['question']
            }],
            ids=[f"doc_{i}"]
        )
        
        # Print progress every 10 documents
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/100 documents added to 'finqa' collection")

    print("\nVector store built successfully! ✅")
    print(f"Collection count: {collection.count()}")

    # Test query for 'revenue'
    print("\n--- Test Query ---")
    query_text = "revenue"
    print(f"Searching for: '{query_text}'")
    
    results = collection.query(
        query_embeddings=[model.encode(query_text).tolist()],
        n_results=1
    )
    
    if results['documents'] and results['documents'][0]:
        first_doc = results['documents'][0][0]
        print(f"Top matched document (first 100 chars):\n{first_doc[:100]}...")
    else:
        print("No documents found.")

if __name__ == "__main__":
    build_store()
