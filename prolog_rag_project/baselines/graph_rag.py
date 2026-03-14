import os
import re
import chromadb
import networkx as nx
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class SimpleGraphRAG:
    def __init__(self):
        print("Initializing Simple-Graph-RAG baseline...")
        self.graph = nx.Graph() 
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
        print("Extracting entities and building graph...")
        all_docs = self.collection.get()
        documents = all_docs['documents']
        ids = all_docs['ids']
        
        number_regex = re.compile(r'\$?[\d\.]+\s*(?:billion|million)', re.IGNORECASE)

        for i, text in enumerate(documents):
            doc_id = ids[i]
            self.graph.add_node(doc_id, type='document', text=text)
            found_numbers = number_regex.findall(text)
            for num in found_numbers:
                num_node = num.lower().strip()
                if not self.graph.has_node(num_node):
                    self.graph.add_node(num_node, type='entity')
                self.graph.add_edge(doc_id, num_node, relation='mentions')

        print(f"Graph Construction Done: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

    def query(self, question, top_k=2):
        print(f"\n--- Processing Query: {question} ---")
        question_embedding = self.encoder.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )
        
        seed_doc_ids = results['ids'][0]
        primary_context = results['documents'][0]
        
        print(f"Seed Documents Found: {seed_doc_ids}")
        
        explored_entities = set()
        bridged_doc_ids = set()
        bridged_contexts = []
        
        for doc_id in seed_doc_ids:
            if self.graph.has_node(doc_id):
                entities = list(self.graph.neighbors(doc_id))
                for ent in entities:
                    explored_entities.add(ent)
                    neighbors = list(self.graph.neighbors(ent))
                    for n_doc in neighbors:
                        if n_doc != doc_id and n_doc not in seed_doc_ids:
                            bridged_doc_ids.add(n_doc)
                            context = self.graph.nodes[n_doc].get('text', '')
                            if context and context not in bridged_contexts:
                                bridged_contexts.append(context)

        print(f"Total Entities Found via Seeds: {len(explored_entities)}")
        print(f"Bridged Documents Found: {len(bridged_doc_ids)}")
        if bridged_doc_ids:
            print(f"Bridged IDs: {list(bridged_doc_ids)[:5]}")

        combined_context = "\n---\n".join(primary_context + bridged_contexts[:2])
        entities_found = ", ".join(list(explored_entities)[:10])
        
        system_prompt = "You are a GraphRAG Assistant. Use primary and bridged documents to answer accurately."
        prompt = f"Question: {question}\n\nEntities: {entities_found}\n\nContext:\n{combined_context}"

        try:
            completion = self.llm.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1
            )
            answer = completion.choices[0].message.content
        except Exception as e:
            answer = f"Error: {str(e)}"

        return {
            'question': question,
            'answer': answer,
            'entities': list(explored_entities),
            'bridged_count': len(bridged_doc_ids),
            'method': 'graph_rag'
        }

if __name__ == "__main__":
    baseline = SimpleGraphRAG()
    # Test query that should trigger bridging (comparing figures)
    res = baseline.query("Compare revenue figures found in the documents")
    print(f"\nFinal Answer: {res['answer'][:500]}...")
    print(f"Entities: {res['entities'][:10]}")
    print(f"Bridged Documents Count: {res['bridged_count']}")
