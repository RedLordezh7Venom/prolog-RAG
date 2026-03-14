import chromadb
from sentence_transformers import SentenceTransformer

class NaiveRAG:
    """
    A baseline RAG system that uses simple semantic retrieval without 
    logical reasoning or formal proofs.
    """
    def __init__(self):
        print("Initializing Naive-RAG baseline...")
        # Load the same embedding model as PrologRAG
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Connect to the same ChromaDB instance
        self.chroma_client = chromadb.PersistentClient(path='./chroma_db')
        
        # Access the 'finqa' collection
        try:
            self.collection = self.chroma_client.get_collection(name="finqa")
            print("Successfully connected to 'finqa' collection.")
        except Exception as e:
            print(f"Error accessing collection: {e}")
            self.collection = None

    def query(self, question, top_k=3):
        """
        Executes a basic retrieve-and-display query.
        """
        if not self.collection:
            return {"error": "Vector database not initialized."}

        # 1. Encode the question
        question_embedding = self.encoder.encode(question).tolist()

        # 2. Retrieve top_k documents
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )

        documents = results['documents'][0]
        
        # 3. Format output
        # For naive baseline, 'answer' is simply the top-ranked snippet
        return {
            'question': question,
            'answer': documents[0] if documents else "No relevant context found.",
            'context': documents,
            'method': 'naive_rag',
            'has_proof': False
        }

if __name__ == "__main__":
    # Quick baseline test
    baseline = NaiveRAG()
    test_q = "What is the profit margin for 2023?"
    result = baseline.query(test_q)
    print(f"\nQuestion: {result['question']}")
    print(f"Answer (Top Snippet): {result['answer'][:200]}...")
    print(f"Has Proof: {result['has_proof']}")
