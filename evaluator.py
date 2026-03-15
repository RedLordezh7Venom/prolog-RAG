import json
import os
import re
from groq import Groq
from dotenv import load_dotenv
from termcolor import colored

load_dotenv()

class RAGEvaluator:
    """
    LLM-as-a-Judge script to grade the Arena results.
    Evaluates accuracy, logic, and numerical reliability.
    """
    def __init__(self):
        self.llm = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def grade_answer(self, question, answer, context):
        prompt = f"""
        You are a financial auditor. Grade the following RAG system response based on accuracy and numerical reliability.
        
        Question: {question}
        
        System Answer: {answer}
        
        Context Snippets: {context[:2000]}
        
        Evaluation Criteria:
        1. Accuracy (0-5): Is the numerical data correct based on the context?
        2. Hallucination (Yes/No): Does the answer contain facts not present in the context?
        3. Logic (0-5): Did the system follow the correct calculation (if applicable)?
        
        Respond in JSON format:
        {{
            "accuracy_score": 0-5,
            "hallucination": "Yes/No",
            "logic_score": 0-5,
            "reason": "Short explanation"
        }}
        """
        
        try:
            completion = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"Error grading: {e}")
            return None

    def evaluate_arena(self, results_path="arena_results.json"):
        with open(results_path, 'r') as f:
            data = json.load(f)
            
        final_scores = []
        
        print(colored("\n--- Starting Evaluation Phase ---", "cyan", attrs=["bold"]))
        
        for q_res in data:
            q_id = q_res['id']
            question = q_res['question']
            print(f"\nGrading Q{q_id}: {question}")
            
            q_eval = {
                "id": q_id,
                "question": question,
                "scores": {}
            }
            
            # Use Prolog-RAG's retrieved docs as context for the judge (usually the most rigorous)
            # or try to consolidate from result if possible. For simplicity here, we assume 
            # the judge can verify based on general context provided in source_docs if available.
            context = ""
            if "Prolog-RAG" in q_res["answers"] and "source_docs" in q_res.get("metadata", {}):
                 context = "\n".join(q_res["metadata"]["source_docs"])
            else:
                 # Fallback: manually fetch context if needed or use the answer itself as proxy (not ideal)
                 context = "Refer to the provided answers to see if they are internally consistent with the query type."

            for model_name, res in q_res["answers"].items():
                grade = self.grade_answer(question, res["answer"], context)
                if grade:
                    q_eval["scores"][model_name] = grade
                    print(f" - {model_name}: Score {grade['accuracy_score']}/5 | Hallucination: {grade['hallucination']}")
            
            final_scores.append(q_eval)
            
        return final_scores

    def save_summary(self, final_scores, filepath="eval_summary.json"):
        with open(filepath, 'w') as f:
            json.dump(final_scores, f, indent=4)
        print(colored(f"\nEvaluation summary saved to {filepath}", "green", attrs=["bold"]))

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    # Note: This requires arena_results.json to be fully populated
    if os.path.exists("arena_results.json"):
        scores = evaluator.evaluate_arena()
        evaluator.save_summary(scores)
    else:
        print("arena_results.json not found. Run arena.py first.")
