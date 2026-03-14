import json
import time
import os
import argparse
from termcolor import colored

# Import all RAG baselines
from prolog_rag_project.baselines.naive_rag import NaiveRAG
from prolog_rag_project.baselines.graph_rag import SOTAGraphRAG
from prolog_rag_project.baselines.corrective_rag import CorrectiveRAG
from prolog_rag_project.baselines.contextual_rag import ContextualRAG
from prolog_rag_project.core.prolog_rag import PrologRAG

class RACArena:
    """
    Unified benchmarking arena to test Prolog-RAG against 4 SOTA baselines.
    """
    def __init__(self):
        print(colored("Initializing RAG Arena...", "cyan", attrs=["bold"]))
        
        # Initialize all systems
        print("Loading Naive RAG...")
        self.naive_rag = NaiveRAG()
        
        print("Loading Graph RAG...")
        self.graph_rag = SOTAGraphRAG()
        
        print("Loading Corrective RAG (CRAG)...")
        self.crag = CorrectiveRAG()
        
        print("Loading Contextual RAG...")
        self.contextual_rag = ContextualRAG()
        
        print(colored("Loading Prolog-RAG (The Challenger)...", "magenta", attrs=["bold"]))
        self.prolog_rag = PrologRAG()
        
        self.models = {
            "Naive": self.naive_rag,
            "Graph": self.graph_rag,
            "CRAG": self.crag,
            "Contextual": self.contextual_rag,
            "Prolog-RAG": self.prolog_rag
        }

    def load_questions(self, filepath="test_questions.json"):
        with open(filepath, 'r') as f:
            return json.load(f)

    def run_benchmark(self, questions, specific_model=None):
        results = []
        
        models_to_run = self.models if not specific_model else {specific_model: self.models[specific_model]}
        
        print(colored(f"\nEvaluating {len(questions)} questions across {len(models_to_run)} models...\n", "yellow"))
        
        for q in questions:
            q_text = q['question']
            q_id = q['id']
            print(colored(f"\n[{q_id}] Q: {q_text}", "white", attrs=["bold"]))
            
            q_results = {
                "id": q_id,
                "question": q_text,
                "type": q['type'],
                "answers": {}
            }
            
            for model_name, model in models_to_run.items():
                start_time = time.time()
                try:
                    # Execute query
                    if model_name == "Prolog-RAG":
                        # Prolog RAG returns a complex dict
                        res = model.query(q_text)
                        answer = res.get('answer', str(res))
                        method = res.get('route', 'UNKNOWN')
                    else:
                        # Baselines return standard dict
                        res = model.query(q_text)
                        answer = res.get('answer', str(res))
                        method = res.get('method', 'UNKNOWN')
                        
                except Exception as e:
                    answer = f"ERROR: {str(e)}"
                    method = "FAILED"
                    
                exec_time = time.time() - start_time
                
                # Store results
                q_results["answers"][model_name] = {
                    "answer": answer,
                    "method": method,
                    "time_sec": round(exec_time, 2)
                }
                
                # Print output
                color = "green" if model_name == "Prolog-RAG" else "blue"
                print(colored(f" └─ {model_name} ({method}) [{exec_time:.1f}s]: ", color) + f"{answer[:150]}...")
                
            results.append(q_results)
            
        return results

    def save_results(self, results, filepath="arena_results.json"):
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        print(colored(f"\nResults saved to {filepath}", "green", attrs=["bold"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG Arena Benchmark")
    parser.add_argument("--model", type=str, help="Specific model to run (Naive, Graph, CRAG, Contextual, Prolog-RAG)")
    parser.add_argument("--question_id", type=int, help="Specific question ID to run")
    args = parser.parse_args()

    arena = RACArena()
    all_questions = arena.load_questions()
    
    if args.question_id:
        all_questions = [q for q in all_questions if q['id'] == args.question_id]
        
    final_results = arena.run_benchmark(all_questions, specific_model=args.model)
    arena.save_results(final_results)
