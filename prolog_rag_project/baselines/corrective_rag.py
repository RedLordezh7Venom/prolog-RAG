import os
import json
import chromadb
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class CorrectiveRAG:
    """
    A SOTA-inspired Corrective RAG (CRAG) implementation.
    Concept: Retrieve -> Grade -> Correct (Search/Rewrite) -> Generate.
    
    This implementation uses:
    1. Retrieval Grader: LLM evaluates document relevance.
    2. Knowledge Refinement: If irrelevant, it triggers a 'fallback' (Query Rewriting).
    3. Final Synthesis: Uses only verified relevant context.
    """
    def __init__(self):
        print("Initializing Corrective-RAG (CRAG) system...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path='./chroma_db')
        self.llm = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        
        try:
            self.collection = self.chroma_client.get_collection(name="finqa")
            print("Successfully connected to 'finqa' collection.")
        except Exception as e:
            print(f"Error accessing collection: {e}")
            self.collection = None

    def _grade_document(self, question, document):
        """
        Retrieval Grader: Evaluates if a document is relevant to the question.
        Returns: 'yes' (relevant) or 'no' (irrelevant).
        """
        prompt = f"""
        You are a grader assessing relevance of a retrieved document to a user question. 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
        It does not need to be a perfect answer, but it must provide useful information for reasoning.

        User question: {question}
        Retrieved document: {document}

        Respond with ONLY a JSON object: {{"relevance": "yes"}} or {{"relevance": "no"}}
        """
        try:
            completion = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"}
            )
            data = json.loads(completion.choices[0].message.content)
            return data.get("relevance", "no").lower()
        except:
            return "no"

    def _rewrite_query(self, question):
        """
        Query Rewriter: Optimizes the question for better retrieval when initial results fail.
        """
        prompt = f"""
        The initial retrieval for the following question failed to find relevant information. 
        Rewrite the question to be more specific, keyword-rich, and optimized for semantic search.
        Keep the original intent.

        Original: {question}
        
        Respond with ONLY the rewritten question string.
        """
        try:
            completion = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.7
            )
            return completion.choices[0].message.content.strip()
        except:
            return question

    def query(self, question, top_k=3):
        print(f"\n--- CRAG-Query: {question} ---")
        
        # 1. Initial Retrieval
        query_embedding = self.encoder.encode(question).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        documents = results['documents'][0]
        
        # 2. Retrieval Grading
        print("Grading retrieved documents...")
        relevant_docs = []
        needs_correction = False
        
        for doc in documents:
            score = self._grade_document(question, doc)
            if score == "yes":
                relevant_docs.append(doc)
            else:
                needs_correction = True

        # 3. Corrective Action: If some docs were irrelevant, try rewriting and re-searching
        if needs_correction and not relevant_docs:
            print("No relevant documents found. Triggering Query Rewriting (Correction)...")
            rewritten_q = self._rewrite_query(question)
            print(f"Rewritten Query: {rewritten_q}")
            
            # Re-retrieval
            new_emb = self.encoder.encode(rewritten_q).tolist()
            new_res = self.collection.query(query_embeddings=[new_emb], n_results=top_k)
            new_docs = new_res['documents'][0]
            
            # Add new docs (we trust the rewriter more for this step)
            relevant_docs.extend(new_docs)
        elif needs_correction:
            print(f"Discarded {len(documents) - len(relevant_docs)} irrelevant docs. Proceeding with {len(relevant_docs)} verified docs.")

        # 4. Final Generation
        context = "\n---\n".join(relevant_docs)
        system_prompt = "You are a specialized financial analyst. Use the verified context provided to answer the question."
        prompt = f"Question: {question}\n\nVerified Context:\n{context}"

        try:
            completion = self.llm.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1
            )
            answer = completion.choices[0].message.content
        except Exception as e:
            answer = f"Error during generation: {e}"

        return {
            'question': question,
            'answer': answer,
            'method': 'corrective_rag',
            'relevant_docs_count': len(relevant_docs),
            'correction_triggered': needs_correction and not (len(relevant_docs) == len(documents))
        }

if __name__ == "__main__":
    crag = CorrectiveRAG()
    # Test question that might be tricky
    test_q = "What is the net income for 2023?"
    res = crag.query(test_q)
    print(f"\nFinal Answer: {res['answer']}")
    print(f"Docs used: {res['relevant_docs_count']}")
    print(f"Correction Triggered: {res['correction_triggered']}")
