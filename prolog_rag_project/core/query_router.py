from enum import Enum
import logging

logger = logging.getLogger(__name__)

class QueryType(Enum):
    PROLOG = "PROLOG"
    VECTOR = "VECTOR"

class QueryRouter:
    """
    Routes queries between formal logical reasoning and semantic search.
    """
    PROLOG_KEYWORDS = [
        "higher", "lower", "compare", "calculate", 
        "margin", "growth", "revenue", "profit"
    ]

    def route(self, question: str) -> QueryType:
        """
        Determines the query type based on keyword matching threshold.
        Returns PROLOG if count >= 2, else VECTOR.
        """
        count = 0
        question_lower = question.lower()
        
        for keyword in self.PROLOG_KEYWORDS:
            if keyword in question_lower:
                count += 1
                logger.debug(f"Prolog keyword '{keyword}' found. Current count: {count}")
        
        if count >= 2:
            return QueryType.PROLOG
        
        return QueryType.VECTOR

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    router = QueryRouter()
    
    test_questions = [
        "What is the profit margin?",
        "What happened in Q3?",
        "Compare revenue of Apple and Microsoft"
    ]
    
    print("--- Testing Query Router ---")
    for q in test_questions:
        decision = router.route(q)
        print(f"Question: '{q}'\nDecision: {decision.value}\n")
