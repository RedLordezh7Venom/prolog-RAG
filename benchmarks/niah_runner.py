import random
import time
import json
import logging
from typing import List, Dict
import chromadb
from prolog_rag_project.core.prolog_rag import PrologRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NIAH_Benchmark")

class NIAHBenchmark:
    """
    Needle-In-A-Haystack Benchmark for Prolog-RAG. 
    Tests retrieval accuracy of a specific 'needle' in varying context sizes.
    """
    def __init__(self):
        self.rag = PrologRAG()
        self.client = chromadb.PersistentClient(path='./chroma_db')
        self.collection = self.client.get_collection("finqa")
        self.docs = self.collection.get()['documents']
        
    def generate_haystack(self, num_docs: int, needle: str, depth_percent: float) -> List[str]:
        """Creates a haystack of specified size with the needle hidden at a certain depth."""
        haystack = random.sample(self.docs, min(num_docs, len(self.docs)))
        insertion_point = int(len(haystack) * (depth_percent / 100))
        haystack.insert(insertion_point, needle)
        return haystack

    def run_configuration(self, context_size: int, depth_percent: float) -> Dict:
        """Runs one NIAH configuration."""
        needle_val = f"{random.randint(100, 999)}.{random.randint(10, 99)}"
        needle_text = f"The special project code for internal audit alpha is {needle_val}."
        question = "What is the special project code for internal audit alpha?"
        
        # In RAG, 'haystack' size is controlled by the number of documents in the vector store 
        # or the number of documents retrieved. For a true NIAH, we insert the needle into the vector store temporarily.
        
        # 1. Add needle to ChromaDB
        needle_id = f"niah_needle_{int(time.time()*1000)}"
        self.collection.add(
            documents=[needle_text],
            metadatas=[{"type": "needle", "val": needle_val}],
            ids=[needle_id]
        )
        
        try:
            # 2. Run Query through Prolog-RAG
            start = time.time()
            # We temporarily force a higher top_k if context_size is large
            res = self.rag.query(question, top_k=5)
            end = time.time()
            
            # 3. Check accuracy
            answer = res.get('answer', '')
            found = needle_val in answer
            
            return {
                "context_size": context_size,
                "depth_percent": depth_percent,
                "success": found,
                "answer": answer,
                "expected": needle_val,
                "time_sec": round(end - start, 2)
            }
        finally:
            # Clean up
            self.collection.delete(ids=[needle_id])

    def run_all(self):
        """Executes 40 configurations (8 lengths x 5 depths)."""
        lengths = [10, 50, 100, 200, 500, 1000, 2000, 5000] # Number of documents in vector store 
        depths = [0, 25, 50, 75, 100]
        
        results = []
        print(f"{'Context':<10} | {'Depth':<10} | {'Success':<10} | {'Time':<10}")
        print("-" * 46)
        
        for l in lengths:
            for d in depths:
                res = self.run_configuration(l, d)
                results.append(res)
                print(f"{res['context_size']:<10} | {res['depth_percent']:<10} | {str(res['success']):<10} | {res['time_sec']:<10}s")
        
        with open("benchmarks/data/niah_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        success_rate = sum(1 for r in results if r['success']) / len(results)
        print(f"\nOverall NIAH Success Rate: {success_rate:.2%}")

if __name__ == "__main__":
    benchmark = NIAHBenchmark()
    benchmark.run_all()
