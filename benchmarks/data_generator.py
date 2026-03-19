import json
import random

def generate_hotpot():
    """Generates 25 multi-hop questions."""
    companies = ['AAL', 'AAP', 'AAPL', 'ABMD', 'APTV', 'AWK', 'BKNG', 'BLL']
    years = [2006, 2007, 2008, 2009]
    questions = []
    
    for i in range(25):
        c1, c2 = random.sample(companies, 2)
        year = random.choice(years)
        questions.append({
            "id": i + 1,
            "question": f"What was the total net income for {c1} and {c2} in {year}?",
            "type": "multi-hop-resoning",
            "required_docs": [f"{c1}/{year}", f"{c2}/{year}"]
        })
    return questions

def generate_frames():
    """Generates 30 numerical reasoning questions."""
    companies = ['AAL', 'AAP', 'AAPL', 'ABMD', 'APTV', 'AWK', 'BKNG', 'BLL']
    years = [2007, 2008, 2013] # We know these years exist
    questions = []
    
    # 1. Growth calculations
    for i in range(15):
        c = random.choice(companies)
        y1, y2 = 2008, 2013
        questions.append({
            "id": i + 1,
            "question": f"By what percentage did {c}'s revenue grow between {y1} and {y2}?",
            "type": "numerical-calculation"
        })
        
    # 2. Comparisons
    for i in range(15):
        c1, c2 = random.sample(companies, 2)
        year = 2007
        questions.append({
            "id": i + 16,
            "question": f"Which company had a higher profit margin in {year}, {c1} or {c2}?",
            "type": "numerical-comparison"
        })
    return questions

def save_all():
    with open("benchmarks/data/hotpot_questions.json", "w") as f:
        json.dump(generate_hotpot(), f, indent=2)
    with open("benchmarks/data/frames_questions.json", "w") as f:
        json.dump(generate_frames(), f, indent=2)
    print("Benchmark data generated successfully in benchmarks/data/")

if __name__ == "__main__":
    save_all()
