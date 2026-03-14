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

    def _extract_answer(self, context_list):
        """
        Heuristic baseline: finds the first sentence in the context that 
        contains a numerical value.
        """
        import re
        # Pattern to match numbers (e.g., 25.0, 100, 1,000)
        number_pattern = re.compile(r'\d+')
        
        for doc in context_list:
            # Simple sentence splitting on periods
            sentences = doc.split('. ')
            for sentence in sentences:
                if number_pattern.search(sentence):
                    return sentence.strip()
                    
        return "Answer not found in context."

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
        
        # 3. Extract heuristic answer
        answer = self._extract_answer(documents)
        
        # 4. Format output
        return {
            'question': question,
            'answer': answer,
            'context': documents,
            'method': 'naive_rag',
            'has_proof': False
        }

if __name__ == "__main__":
    # Quick baseline test
    baseline = NaiveRAG()
    test_q = "What is the profit margin?"
    result = baseline.query(test_q)
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Has Proof: {result['has_proof']}")
