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

    def translate_to_prolog(self, question):
        """
        Simple keyword-based translation from NL to Prolog query.
        """
        question_lower = question.lower()
        if 'margin' in question_lower:
            return "profit_margin(DocId, M)"
        if 'growth' in question_lower:
            return "growth_rate(DocOld, DocNew, Rate)"
        return None

    def _format_answer(self, question, results):
        """
        Formats Prolog reasoning results into a human-readable answer.
        """
        if not results:
            return None
            
        res = results[0]
        if 'M' in res:
            return f"Profit margin: {res['M']}%"
        if 'Rate' in res:
            return f"Growth rate: {res['Rate']}%"
        
        # Generic formatting for other potential fields
        parts = [f"{k}: {v}" for k, v in res.items() if k not in ['DocId', 'DocOld', 'DocNew']]
        return ", ".join(parts) if parts else "Condition met in Knowledge Base."

    def query(self, question, top_k=3):
        """
        Executes a query through the Prolog-RAG pipeline.
        """
        print(f"\n--- Query: {question} ---")
        
        # 1. Route the query
        query_type = self.router.route(question)
        print(f"Routing Decision: {query_type.value}")

        # 2. Vector Retrieval
        print(f"Retrieving top {top_k} relevant documents...")
        question_embedding = self.encoder.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        ids = results['ids'][0]
        print(f"Retrieved {len(documents)} documents.")

        # 3. Fact Extraction & KB Loading
        prolog_results = []
        proof_trace = []

        if query_type == QueryType.PROLOG:
            print("Prolog path detected. Extracting facts...")
            self.prolog_kb.clear() # Clear KB for new query reasoning
            
            total_facts = 0
            for doc_text, doc_meta, doc_id in zip(documents, metadatas, ids):
                actual_id = doc_meta.get('finqa_id', doc_id)
                facts = self.fact_extractor.extract_from_text(doc_text, actual_id)
                
                for fact in facts:
                    if self.prolog_kb.add_fact(fact):
                        total_facts += 1
            
            print(f"Loaded {total_facts} facts into Prolog KB.")

            # 4. Prolog Reasoning
            prolog_query = self.translate_to_prolog(question)
            if prolog_query:
                print(f"Translated Prolog Query: {prolog_query}")
                prolog_results, proof_trace = self.prolog_kb.query(prolog_query)
                print(f"Reasoning Results: {prolog_results}")
            else:
                print("Could not translate question to Prolog query.")
        
        # 5. Final Answer Generation
        answer = self._format_answer(question, prolog_results)
        
        # Fallback to vector search snippet if no prolog answer
        if not answer:
            print("Falling back to vector snippet...")
            answer = f"Based on retrieved documents: {documents[0][:200]}..."

        return {
            'question': question,
            'route': query_type.value,
            'answer': answer,
            'proof_trace': proof_trace,
            'source_docs': documents[:2]
        }

if __name__ == "__main__":
    import json
    # Test initialization and routing
    rag = PrologRAG()
    
    result = rag.query("What is the profit margin?")
    print("\n--- FULL RESULT DICT ---")
    print(json.dumps(result, indent=2))
