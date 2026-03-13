import logging

logger = logging.getLogger(__name__)

class QueryRouter:
    """
    Decides whether to route a query to the Prolog reasoning engine or the standard Vector RAG path.
    """
    def __init__(self, threshold=2):
        self.threshold = threshold
        self.prolog_keywords = {
            'higher': 1, 'lower': 1, 'more': 1, 'less': 1,
            'compare': 2, 'difference': 1,
            'calculate': 2, 'margin': 2, 'growth': 2,
            'all': 1, 'when': 1, 'first': 1,
            'percent': 1, 'percentage': 1, 'total': 1,
            'increase': 1, 'decrease': 1
        }

    def route(self, query):
        """
        Analyzes the query and returns 'PROLOG' or 'VECTOR'.
        """
        score = 0
        query_lower = query.lower()
        
        for keyword, weight in self.prolog_keywords.items():
            if keyword in query_lower:
                score += weight
        
        decision = 'PROLOG' if score >= self.threshold else 'VECTOR'
        logger.info(f"Query: '{query}' | Score: {score} | Decision: {decision}")
        return decision
