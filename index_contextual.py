import os
import json
import chromadb
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

def generate_contextual_summary(text, client, model="llama-3.3-70b-versatile"):
    """
    Generates a brief document-level context for a chunk.
    As per Anthropic's blog: 50-100 words summarizing the document context.
    """
    prompt = f"Please provide a one-sentence context for the following financial document snippet to clarify its subject and scope (e.g., 'This document is about Apple's 2023 revenue results').\n\nText: {text[:1000]}"
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating context: {e}")
        return ""

def index_contextual_data():
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chroma_client = chromadb.PersistentClient(path='./chroma_db')
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load original collection
    try:
        original_col = chroma_client.get_collection(name="finqa")
        data = original_col.get()
        documents = data['documents']
        ids = data['ids']
        metadatas = data['metadatas']
    except Exception as e:
        print(f"Error loading original collection: {e}")
        return

    # Create/Load Contextual collection
    contextual_col = chroma_client.get_or_create_collection(name="finqa_contextual")
    
    print(f"Contextualizing {len(documents)} document chunks...")
    
    contextual_docs = []
    contextual_metadatas = []
    contextual_ids = []
    
    # For speed in development, we'll only do a subset or check if already indexed
    existing_count = len(contextual_col.get()['ids'])
    if existing_count >= len(documents):
        print("Contextual collection already appears to be populated. Skipping indexing.")
        return

    for i in range(len(documents)):
        doc_text = documents[i]
        doc_id = ids[i]
        
        # Generate context (ideally we do this once per document, but here chunks are documents in finqa subset)
        context = generate_contextual_summary(doc_text, client)
        contextualized_text = f"Context: {context}\n\nContent: {doc_text}"
        
        emb = encoder.encode(contextualized_text).tolist()
        
        contextual_col.add(
            ids=[doc_id],
            embeddings=[emb],
            documents=[contextualized_text],
            metadatas=[{**metadatas[i], "original_id": doc_id, "context": context}]
        )
        if (i+1) % 10 == 0:
            print(f"Indexed {i+1}/{len(documents)} docs...")

    print("Indexing complete.")

if __name__ == "__main__":
    index_contextual_data()
