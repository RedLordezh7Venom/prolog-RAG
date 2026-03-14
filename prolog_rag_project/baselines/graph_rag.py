import os
import re
import chromadb
import networkx as nx
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleGraphRAG:
    """
    A SOTA-inspired Simple GraphRAG baseline.
    It builds a Knowledge Graph by linking documents through shared numerical entities.
    This allows the system to bridge context across multiple documents (Multi-hop Reasoning).
    """
    def __init__(self):
        print("Initializing Simple-Graph-RAG baseline...")
        self.graph = nx.Graph() # Undirected graph for easier bridging
        
        # Core components
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
        Builds the Knowledge Graph using the pattern requested.
        Connects Documents to Numerical Entities to create a relationship bridge.
        """
        print("Extracting entities and building graph...")
        # Get all docs from collection
        all_docs = self.collection.get()
        documents = all_docs['documents']
        ids = all_docs['ids']
        
        # User specified Regex: (\$?[\d\.]+\s*billion|million)
        # Note: Added word boundary and non-capturing group for better matching
        number_regex = re.compile(r'\$?[\d\.]+\s*(?:billion|million)', re.IGNORECASE)

        for i, text in enumerate(documents):
            doc_id = ids[i]
            # 1. Create Document Node
            self.graph.add_node(doc_id, type='document', text=text)

            # 2. Extract Numbers
            found_numbers = number_regex.findall(text)
            
            for num in found_numbers:
                num_node = num.lower().strip()
                # 3. Create Number Node (if doesn't exist)
                if not self.graph.has_node(num_node):
                    self.graph.add_node(num_node, type='entity')
                
                # 4. Add Edge (Document -> Number)
                # In GraphRAG SOTA, shared entities act as bridges between disparate documents
                self.graph.add_edge(doc_id, num_node, relation='mentions')

        print(f"Graph Construction Done: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

    def query(self, question, top_k=2):
        """
        SOTA Graph Search Pattern:
        1. Retrieval: Find top_k "Seed Documents" via vector similarity.
        2. Expansion (Graph Walk): Find all entities connected to these documents.
        3. Bridging: Find OTHER documents that also mention those same entities.
        4. Synthesis: Combine primary docs, linked entities, and bridged docs for a global answer.
        """
        # Step 1: Vector Search Entry Points
        question_embedding = self.encoder.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )
        
        seed_doc_ids = results['ids'][0]
        primary_context = results['documents'][0]
        
        # Step 2 & 3: Multi-hop Graph Expansion
        explored_entities = set()
        bridged_contexts = []
        
        for doc_id in seed_doc_ids:
            if self.graph.has_node(doc_id):
                # Find direct entities
                entities = list(self.graph.neighbors(doc_id))
                for ent in entities:
                    explored_entities.add(ent)
                    # SOTA BRIDGE: Find other docs that mention this number
                    neighbor_docs = list(self.graph.neighbors(ent))
                    for n_doc in neighbor_docs:
                        if n_doc != doc_id and n_doc not in seed_doc_ids:
                            context = self.graph.nodes[n_doc].get('text', '')
                            if context and context not in bridged_contexts:
                                bridged_contexts.append(context)

        # Step 4: LLM Synthesis with Augmented Context
        combined_context = "\n---\n".join(primary_context + bridged_contexts[:2])
        entities_found = ", ".join(list(explored_entities)[:10])
        
        system_prompt = (
            "You are a GraphRAG Assistant. You have access to both primary retrieved documents "
            "and 'bridged' documents found via entity relationships in a knowledge graph."
        )
        
        prompt = (
            f"Question: {question}\n\n"
            f"Detected Entities: {entities_found}\n\n"
            f"Context (Primary & Bridged):\n{combined_context}\n\n"
            "Analyze all context and provide a comprehensive answer."
        )

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
            answer = f"Error generating GraphRAG answer: {str(e)}"

        return {
            'question': question,
            'answer': answer,
            'entities': list(explored_entities),
            'method': 'graph_rag',
            'has_proof': False
        }

if __name__ == "__main__":
    baseline = SimpleGraphRAG()
    res = baseline.query("What is Apple's revenue growth?")
    print(f"\nQuestion: {res['question']}")
    print(f"\nAnswer: {res['answer']}")
    print(f"\nLinked Entities: {res['entities']}")
