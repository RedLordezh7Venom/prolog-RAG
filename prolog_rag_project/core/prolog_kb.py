from pyswip import Prolog
import logging

logger = logging.getLogger(__name__)

class PrologKnowledgeBase:
    """
    Manages the SWI-Prolog knowledge base and executes reasoning queries.
    """
    def __init__(self):
        self.prolog = Prolog()
        self.facts_loaded = []
        self._load_rules()

    def _load_rules(self):
        """Initialize standard financial reasoning rules."""
        rules = [
            # Profit margin calculation: profit_margin(DocId, Margin)
            "profit_margin(DocId, Margin) :- revenue(DocId, Rev), net_income(DocId, NI), Rev > 0, Margin is (NI / Rev) * 100",
            
            # Revenue comparison: higher_revenue(Doc1, Doc2)
            "higher_revenue(Doc1, Doc2) :- revenue(Doc1, Rev1), revenue(Doc2, Rev2), Rev1 > Rev2",
            
            # Growth rate calculation: growth_rate(DocOld, DocNew, Rate)
            "growth_rate(DocOld, DocNew, Rate) :- revenue(DocOld, RevOld), revenue(DocNew, RevNew), RevOld > 0, Rate is ((RevNew - RevOld) / RevOld) * 100",
            
            # Constraint filtering: meets_criteria(Company, MinRevenue, MinMargin)
            "meets_criteria(Company, MinRevenue, MinMargin) :- revenue(Company, Rev), Rev >= MinRevenue, profit_margin(Company, Margin), Margin >= MinMargin",
            
            # Temporal: first time condition met
            "first_exceeds(Company, Threshold, Year) :- revenue(Company, Year, Rev), Rev >= Threshold, \+ (revenue(Company, YearBefore, RevBefore), YearBefore < Year, RevBefore >= Threshold)"
        ]
        for rule in rules:
            self.prolog.assertz(rule)
            logger.debug(f"Asserted rule: {rule}")

    def add_fact(self, fact_str):
        """
        Adds a pre-formatted fact string to the KB.
        Returns True if successful, False otherwise.
        """
        try:
            self.prolog.assertz(fact_str)
            self.facts_loaded.append(fact_str)
            logger.debug(f"Added fact: {fact_str}")
            return True
        except Exception as e:
            logger.error(f"Error adding fact '{fact_str}': {e}")
            return False

    def query(self, query_str):
        """
        Executes a Prolog query.
        Returns tuple: (list of result dicts, proof_trace list)
        """
        try:
            logger.debug(f"Executing query: {query_str}")
            results = list(self.prolog.query(query_str))
            return results, []
        except Exception as e:
            logger.error(f"Error executing query '{query_str}': {e}")
            return [], []

    def clear(self):
        """Clears the dynamic facts from the KB (simple reset simulation)."""
        # Note: pyswip doesn't have a direct 'clear all', so we might need a more robust reset 
        # for a production system. For now, we'll re-init if needed.
        self.prolog = Prolog()
        self.facts_loaded = []
        self._load_rules()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kb = PrologKnowledgeBase()
    
    # Growth Rate Test Scenario
    print("--- Testing Growth Rate ---")
    kb.add_fact("revenue('old', 100)") # 100 million
    kb.add_fact("revenue('new', 120)") # 120 million
    
    results, proof = kb.query("growth_rate('old', 'new', G)")
    if results:
        growth = results[0]['G']
        print(f"Revenue Growth Rate: {growth}% ✅")
    else:
        print("Growth calculation failed. ❌")
