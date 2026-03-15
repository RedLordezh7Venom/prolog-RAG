import os
import re
from enum import Enum
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class QueryType(Enum):
    PROLOG = "PROLOG"
    VECTOR = "VECTOR"

class QueryRouter:
    """
    Routes queries and translates natural language to formal Prolog queries.
    """
    PROLOG_KEYWORDS = [
        "higher", "lower", "compare", "calculate", 
        "margin", "growth", "revenue", "profit",
        "total", "exceed", "first", "target"
    ]

    def __init__(self):
        self.llm = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        
        self.translation_prompt = """
        You are a Natural Language to Prolog translator.
        Translate the user's financial question into a single valid SWI-Prolog query.
        
        Available predicates in the Knowledge Base:
        - revenue(Company, Year, AmountInMillions).
        - net_income(Company, Year, AmountInMillions).
        - profit_margin_target(Company, Year, Percentage).
        - growth_target(Company, Year, Percentage).
        
        Available KB Rules:
        - profit_margin(DocId, Margin): Calculates margin if both revenue and net_income exist for a doc.
        - growth_rate(DocOld, DocNew, Rate): Calculates revenue growth between two docs.
        
        Rules for translation:
        1. Output ONLY the raw SWI-Prolog query string. No markdown, no explanations.
        2. Variables must start with an Uppercase letter (e.g., DocId, X, Margin).
        3. Atoms (companies, specific years if used as IDs) must be lowercase (e.g., apple, 2023).
        4. End the query without a period (the system handles it).
        
        Examples:
        Q: "What is the profit margin?" -> profit_margin(DocId, Margin)
        Q: "Did revenue exceed 100000?" -> revenue(DocId, Year, Rev), Rev > 100000
        Q: "What is apple's 2023 revenue?" -> revenue(apple, 2023, Rev)
        
        Question: {question}
        Prolog Query:
        """

    def route(self, question: str) -> QueryType:
        """
        Determines the query type based on keyword matching threshold.
        Returns PROLOG if count >= 1, else VECTOR.
        """
        count = 0
        question_lower = question.lower()
        
        for keyword in self.PROLOG_KEYWORDS:
            if keyword in question_lower:
                count += 1
                logger.debug(f"Prolog keyword '{keyword}' found. Current count: {count}")
        
        if count >= 1:
            return QueryType.PROLOG
        
        return QueryType.VECTOR

    def translate_to_prolog(self, question: str) -> str:
        """
        Uses LLM to translate natural language into a Prolog query.
        """
        try:
            prompt = self.translation_prompt.format(question=question)
            completion = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0
            )
            
            raw_query = completion.choices[0].message.content.strip()
            # Clean up potential markdown formatting if the LLM ignores instructions
            raw_query = re.sub(r'```(\w+)?', '', raw_query).strip('` \n.')
            return raw_query
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return ""

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    router = QueryRouter()
    
    test_questions = [
        "What is the profit margin?",
        "What happened in Q3?",
        "Compare revenue of Apple and Microsoft"
    ]
    
    print("--- Testing Query Router & Translator ---")
    for q in test_questions:
        decision = router.route(q)
        print(f"Question: '{q}'\nDecision: {decision.value}")
        if decision == QueryType.PROLOG:
            pl_query = router.translate_to_prolog(q)
            print(f"Translated Prolog: {pl_query}")
        print("-" * 30)
