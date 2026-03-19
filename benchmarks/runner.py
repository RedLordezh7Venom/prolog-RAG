import json
import os
import time
import argparse
from prolog_rag_project.core.prolog_rag import PrologRAG
from termcolor import colored

class BenchmarkRunner:
    def __init__(self):
        self.rag = PrologRAG()
        self.results_dir = "benchmarks/results"
        os.makedirs(self.results_dir, exist_ok=True)

    def run_suite(self, name, questions):
        print(colored(f"\n--- Running Benchmark Suite: {name} ({len(questions)} questions) ---", "cyan", attrs=["bold"]))
        results = []
        
        for q in questions:
            q_text = q['question']
            print(f"[{q['id']}] Q: {q_text}")
            
            start = time.time()
            try:
                res = self.rag.query(q_text)
                end = time.time()
                
                results.append({
                    "id": q['id'],
                    "question": q_text,
                    "answer": res.get('answer', ''),
                    "route": res.get('route', 'UNKNOWN'),
                    "time_sec": round(end - start, 2),
                    "context": res.get('source_docs', [])
                })
                print(f"  └─ Success ({res.get('route')}) in {end-start:.1f}s")
            except Exception as e:
                print(f"  └─ FAILED: {str(e)}")
                
        output_file = f"{self.results_dir}/{name.lower().replace(' ', '_')}_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(colored(f"\nSuite {name} completed. Results saved to {output_file}", "green"))

    def run_all(self):
        # 1. Custom Financial QA (10)
        with open("test_questions.json", "r") as f:
            custom_q = json.load(f)
        self.run_suite("Custom Financial QA", custom_q)

        # 2. HotpotQA (25)
        with open("benchmarks/data/hotpot_questions.json", "r") as f:
            hotpot_q = json.load(f)
        self.run_suite("HotpotQA", hotpot_q)

        # 3. FRAMES Numerical (30)
        with open("benchmarks/data/frames_questions.json", "r") as f:
            frames_q = json.load(f)
        self.run_suite("FRAMES Numerical", frames_q)

if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_all()
