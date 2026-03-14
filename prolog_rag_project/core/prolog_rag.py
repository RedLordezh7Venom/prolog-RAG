import chromadb
from sentence_transformers import SentenceTransformer
from .prolog_kb import PrologKnowledgeBase
from .fact_extractor import FinancialFactExtractor
from .query_router import QueryRouter, QueryType

class PrologRAG:
    """
    Main pipeline for Prolog-RAG: Financial Question Answering with Explainable Logical Reasoning.
    """
    def __init__(self):
        print("Initializing Prolog-RAG components...")
        
        # 1. Initialize Sentence Transformer for embeddings
        print(" - Loading embedding model (all-MiniLM-L6-v2)...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Initialize ChromaDB client
        print(" - Connecting to ChromaDB (./chroma_db)...")
        self.chroma_client = chromadb.PersistentClient(path='./chroma_db')
        
        # 3. Get existing 'finqa' collection
        print(" - Accessing 'finqa' collection...")
        self.collection = self.chroma_client.get_collection(name="finqa")
        
        # 4. Initialize Prolog Knowledge Base
        print(" - Loading Prolog Knowledge Base...")
        self.prolog_kb = PrologKnowledgeBase()
        
        # 5. Initialize Fact Extractor
        print(" - Initializing Financial Fact Extractor...")
        self.fact_extractor = FinancialFactExtractor()
        
        # 6. Initialize Query Router
        print(" - Setting up Query Router...")
        self.router = QueryRouter()
        
        print("\nProlog-RAG system initialized successfully! ✅")

    def query(self, question, top_k=3):
        """
        Executes a query through the Prolog-RAG pipeline.
        """
        print(f"\n--- Query: {question} ---")
        
        # 1. Route the query
        query_type = self.router.route(question)
        print(f"Routing Decision: {query_type.value}")
        
        return query_type

if __name__ == "__main__":
    # Test initialization and routing
    rag = PrologRAG()
    
    rag.query("What is the profit margin of Apple?")
    rag.query("Compare revenue and profit growth for 2023")
    rag.query("What is Apple's mission statement?")
