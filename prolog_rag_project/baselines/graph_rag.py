import os
import re
import json
import chromadb
import networkx as nx
from networkx.algorithms import community
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SOTAGraphRAG:
    """
    A SOTA-inspired GraphRAG implementation featuring:
    1. LLM-based Entity & Relation Extraction (Triplets)
    2. Hierarchical Community Summarization (Microsoft Method)
    3. Hybrid Search (Global Community Summaries + Local Entity Bridges)
    """
    def __init__(self, limit_docs=20):
        print("Initializing SOTA-Graph-RAG...")
        self.graph = nx.Graph()
        self.community_summaries = {}
        self.limit_docs = limit_docs # Limit for indexing speed in demo
        
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
            self._build_communities()

    def _extract_triplets_llm(self, text):
        """
        Uses LLM to extract entities and relations (triplets) from text.
        """
        prompt = f"""
        Extract key financial entities and their relationships from the following text.
        Format the output as a JSON object with:
        "entities": list of names (e.g. ["Apple", "$394 billion"])
        "triplets": list of [subject, relation, object] (e.g. [["Apple", "reported_revenue", "$394 billion"]])
        
        Text: {text[:2000]}
        
        JSON:
        """
        try:
            completion = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"Extraction error: {e}")
            return {"entities": [], "triplets": []}

    def _build_graph(self):
        """
        Builds the graph using LLM-extracted entities and relations.
        """
        print(f"Building Knowledge Graph (Processing first {self.limit_docs} docs)...")
        all_docs = self.collection.get()
        documents = all_docs['documents'][:self.limit_docs]
        ids = all_docs['ids'][:self.limit_docs]

        for i, text in enumerate(documents):
            doc_id = ids[i]
            self.graph.add_node(doc_id, type='document', text=text)

            # LLM Extraction
            print(f" - Extraction for {doc_id}...")
            data = self._extract_triplets_llm(text)
            
            # Add Entity Nodes
            for entity in data.get('entities', []):
                ent_node = str(entity).lower().strip()
                if not self.graph.has_node(ent_node):
                    self.graph.add_node(ent_node, type='entity')
                self.graph.add_edge(doc_id, ent_node, relation='mentions')
            
            # Add Triplets as Edges between entities
            for sub, rel, obj in data.get('triplets', []):
                sub_n = str(sub).lower().strip()
                obj_n = str(obj).lower().strip()
                
                if not self.graph.has_node(sub_n): self.graph.add_node(sub_n, type='entity')
                if not self.graph.has_node(obj_n): self.graph.add_node(obj_n, type='entity')
                
                self.graph.add_edge(sub_n, obj_n, relation=rel)

        print(f"Graph Built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

    def _build_communities(self):
        """
        Groups nodes into communities and generates summaries for each.
        """
        print("Detecting Communities & Generating Summaries...")
        # Use simple modularity-based communities
        comm_list = list(community.greedy_modularity_communities(self.graph))
        
        for i, comm in enumerate(comm_list):
            if len(comm) < 3: continue # Skip tiny communities
            
            # Collect context for the community
            comm_nodes = list(comm)
            comm_text = ""
            for node in comm_nodes:
                if self.graph.nodes[node].get('type') == 'document':
                    comm_text += self.graph.nodes[node].get('text', '')[:500] + "\n"
                else:
                    comm_text += f"Entity: {node}\n"
            
            # Summarize Community with LLM
            prompt = f"Summarize the key financial information in this cluster of entities and documents:\n{comm_text[:3000]}"
            try:
                completion = self.llm.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=0.1
                )
                self.community_summaries[i] = completion.choices[0].message.content
                print(f" - Community {i} summarized.")
            except:
                continue

    def query(self, question, top_k=2):
        """
        SOTA Hybrid Search:
        1. Local Search: Find entities via Vector Search -> Graph Expansion.
        2. Global Search: Rank Community Summaries by relevance to question.
        3. Synthesis: Final answer using both Local & Global context.
        """
        print(f"\n--- SOTA-Query: {question} ---")
        
        # 1. Local Search (Entity-Bridge)
        question_embedding = self.encoder.encode(question).tolist()
        results = self.collection.query(query_embeddings=[question_embedding], n_results=top_k)
        seed_doc_ids = results['ids'][0]
        
        local_context = results['documents'][0]
        bridged_ids = set()
        for doc_id in seed_doc_ids:
            if self.graph.has_node(doc_id):
                neighbors = list(self.graph.neighbors(doc_id))
                for node in neighbors:
                    # Find other docs sharing this entity/triplet
                    bridges = list(self.graph.neighbors(node))
                    for b_doc in bridges:
                        if b_doc != doc_id and self.graph.nodes[b_doc].get('type') == 'document':
                            bridged_ids.add(b_doc)

        bridged_text = [self.graph.nodes[bid]['text'] for bid in list(bridged_ids)[:2]] if bridged_ids else []

        # 2. Global Search (Community relevance)
        # Simply use the LLM to pick relevant summaries (or we could use embeddings)
        all_summaries = "\n".join([f"Comm {i}: {s[:300]}..." for i, s in self.community_summaries.items()])
        prompt = f"Given these community summaries, which one is most relevant to: '{question}'? Just return the number.\n{all_summaries}"
        
        try:
            resp = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0
            ).choices[0].message.content
            # Extract number
            comm_id = int(re.search(r'\d+', resp).group()) if re.search(r'\d+', resp) else None
            relevant_summary = self.community_summaries.get(comm_id, "")
        except:
            relevant_summary = ""

        # 3. Synthesis
        final_prompt = f"""
        Question: {question}
        
        Local Context (Primary Docs): {local_context}
        Bridged Context (Related Docs): {bridged_text}
        Global Context (Community Summary): {relevant_summary}
        
        Synthesize a final, high-fidelity answer:
        """
        
        answer = self.llm.chat.completions.create(
            messages=[{"role": "user", "content": final_prompt}],
            model=self.model,
            temperature=0.1
        ).choices[0].message.content

        return {
            'question': question,
            'answer': answer,
            'method': 'sota_graph_rag',
            'communities': len(self.community_summaries)
        }

if __name__ == "__main__":
    # Note: Using small limit_docs for fast demonstration
    sota_rag = SOTAGraphRAG(limit_docs=5)
    res = sota_rag.query("What is Apple's revenue and how does it relate to other entities?")
    print(f"\nFinal Answer:\n{res['answer']}")
