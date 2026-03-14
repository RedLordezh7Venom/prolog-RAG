import os
import re
import chromadb
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleGraphRAG:
    """
    A baseline GraphRAG system that builds a relationship graph between 
    documents and the numerical figures they contain.
    """
    def __init__(self):
        print("Initializing Simple-Graph-RAG baseline...")
        self.graph = nx.DiGraph()
        
        # Initialize components for retrieval and synthesis
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
            
        if self.collection:
            self._build_graph()

    def _build_graph(self):
        """
        Fetches all documents and builds a graph connecting documents to numbers.
        """
        print("Building Knowledge Graph...")
        # Get all documents from ChromaDB
        all_docs = self.collection.get()
        documents = all_docs['documents']
        ids = all_docs['ids']
        metadatas = all_docs['metadatas']

        # Regex to extract numbers like $394.3 billion or 500 million
        number_regex = re.compile(r'\$?[\d\.]+\s*(?:billion|million)', re.IGNORECASE)

        for i, doc_text in enumerate(documents):
            doc_id = ids[i]
            # Add Document Node
            self.graph.add_node(doc_id, type='document', text=doc_text)

            # Find all numbers in the text
            found_numbers = number_regex.findall(doc_text)
            
            for num in found_numbers:
                # Add Number Node (normalized to lowercase for consistency)
                num_node = num.lower().strip()
                if not self.graph.has_node(num_node):
                    self.graph.add_node(num_node, type='number')
                
                # Add Edge representing 'contains' relationship
                self.graph.add_edge(doc_id, num_node, relation='contains')

        print(f"Graph Build Complete: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

    def query(self, question, top_k=3):
        """
        Simple Graph-based traversal:
        1. Find relevant doc nodes via vector search.
        2. Find all number nodes connected to those docs.
        3. Synthesize answer using docs + linked numbers.
        """
        # 1. Vector Search for entry points
        question_embedding = self.encoder.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )
        
        seed_doc_ids = results['ids'][0]
        context_parts = []
        linked_data = []

        # 2. Traverse Graph
        for doc_id in seed_doc_ids:
            if self.graph.has_node(doc_id):
                doc_text = self.graph.nodes[doc_id].get('text', '')
                context_parts.append(doc_text)
                
                # Get neighbors (numbers found in this doc)
                numbers = list(self.graph.neighbors(doc_id))
                linked_data.extend(numbers)

        # 3. LLM Synthesis
        context_str = "\n\n".join(context_parts)
        entities_str = ", ".join(set(linked_data))
        
        system_prompt = "You are a financial analyst using a Knowledge Graph. Answer based on the context and linked entities."
        prompt = f"Question: {question}\n\nRetrieved Context:\n{context_str}\n\nLinked Numerical Entities in Graph: {entities_str}"

        try:
            completion = self.llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.1
            )
            answer = completion.choices[0].message.content
        except Exception as e:
            answer = f"Error during LLM synthesis: {str(e)}"

        return {
            'question': question,
            'answer': answer,
            'linked_entities': list(set(linked_data)),
            'method': 'graph_rag',
            'has_proof': False
        }

if __name__ == "__main__":
    # Test the GraphRAG
    grag = SimpleGraphRAG()
    test_q = "What are the key financial figures mentioned?"
    res = grag.query(test_q)
    print(f"\nQuestion: {res['question']}")
    print(f"Answer: {res['answer'][:300]}...")
    print(f"Linked Entities: {res['linked_entities']}")
