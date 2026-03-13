from pyswip import Prolog
import logging

logger = logging.getLogger(__name__)

class PrologKB:
    """
    Manages the SWI-Prolog knowledge base and executes reasoning queries.
    """
    def __init__(self):
        self.prolog = Prolog()
        self._init_rules()

    def _init_rules(self):
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

    def add_fact(self, predicate, *args):
        """
        Adds a fact to the KB.
        Example: add_fact("revenue", "apple_2023", 394300000000)
        """
        fact = f"{predicate}({', '.join(map(str, args))})"
        self.prolog.assertz(fact)
        logger.debug(f"Added fact: {fact}")

    def query(self, query_str):
        """
        Executes a Prolog query and returns results as a list of dictionaries.
        """
        logger.debug(f"Executing query: {query_str}")
        return list(self.prolog.query(query_str))

    def clear(self):
        """Clears the dynamic facts from the KB (simple reset simulation)."""
        # Note: pyswip doesn't have a direct 'clear all', so we might need a more robust reset 
        # for a production system. For now, we'll re-init if needed.
        self.prolog = Prolog()
        self._init_rules()
