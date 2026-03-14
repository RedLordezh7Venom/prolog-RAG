from prolog_rag_project.core.prolog_rag import PrologRAG
import json

rag = PrologRAG()
result = rag.query("What is the profit margin?")
print("\n--- LLM ANSWER ---")
print(result['answer'])
