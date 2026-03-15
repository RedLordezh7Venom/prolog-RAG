import os
import chromadb
from groq import Groq
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

class ContextualRAG:
    """
    Implementation of Anthropic's 'Contextual Retrieval' architecture.
    Features:
    1. Contextual Embeddings: Chunks augmented with document-level summaries.
    2. Contextual BM25: Lexical matching on augmented chunks.
    3. Hybrid Search: Combination of Vector and BM25 results using RRF.
    """
    def __init__(self):
        print("Initializing Contextual-RAG system...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path='./chroma_db')
        self.llm = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        
        try:
            # Use the pre-indexed contextual collection
            self.collection = self.chroma_client.get_collection(name="finqa_contextual")
            print("Successfully connected to 'finqa_contextual' collection.")
        except Exception as e:
            print(f"Error accessing contextual collection: {e}. Ensure index_contextual.py has run.")
            self.collection = None

        if self.collection:
            self._build_bm25_index()

    def _build_bm25_index(self):
        """
        Builds a lexical index for BM25 search.
        """
        print("Building BM25 index on contextualized corpus...")
        data = self.collection.get()
        self.documents = data['documents']
        self.ids = data['ids']
        
        # Tokenize documents for BM25
        tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index built with {len(self.documents)} documents.")

    def _rrf(self, vector_results, bm25_results, k=60):
        """
        Reciprocal Rank Fusion (RRF) to combine results from multiple retrieval paths.
        """
        scores = {}
        
        # vector_results is a list of ids
        for rank, doc_id in enumerate(vector_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)
            
        # bm25_results is a list of ids
        for rank, doc_id in enumerate(bm25_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)
            
        # Re-sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_ids

    def query(self, question, top_k=3):
        print(f"\n--- Contextual-Hybrid-Query: {question} ---")
        
        if not self.collection:
            return {"error": "Index not initialized."}

        # 1. Path A: Vector Search (Semantic)
        query_embedding = self.encoder.encode(question).tolist()
        v_results = self.collection.query(query_embeddings=[query_embedding], n_results=10)
        v_ids = v_results['ids'][0]
        
        # 2. Path B: BM25 Search (Lexical)
        tokenized_query = question.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Get top 10 indices
        top_n_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
        bm25_ids = [self.ids[i] for i in top_n_indices]

        # 3. Hybrid Merge (RRF)
        fused_ids = self._rrf(v_ids, bm25_ids)
        final_ids = fused_ids[:top_k]
        
        # Fetch actual content for top candidates
        final_docs = []
        for doc_id in final_ids:
            doc_idx = self.ids.index(doc_id)
            final_docs.append(self.documents[doc_idx])

        # 4. Final Synthesis
        context = "\n---\n".join(final_docs)
        system_prompt = "You are a specialized financial analyst. Use the provided contextualized context to answer accurately."
        prompt = f"Question: {question}\n\nContext:\n{context}"

        try:
            completion = self.llm.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1
            )
            answer = completion.choices[0].message.content
        except Exception as e:
            answer = f"Error: {e}"

        return {
            'question': question,
            'answer': answer,
            'method': 'contextual_rag',
            'ids_retrieved': final_ids,
            'has_proof': False
        }

if __name__ == "__main__":
    crag = ContextualRAG()
    test_q = "What is the net sales figure for 2011?"
    res = crag.query(test_q)
    print(f"\nFinal Answer: {res['answer']}")
    print(f"Docs used: {res['ids_retrieved']}")
