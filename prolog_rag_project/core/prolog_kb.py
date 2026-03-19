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
        # Ensure predicates are declared as dynamic to allow runtime updates and prevent existence errors
        # Using comma-separated dynamic declarations for efficiency
        try:
            list(self.prolog.query("dynamic([revenue/3, operating_income/3, net_income/3, cost_of_sales/3, allocated_cost/4])"))
            
            # Clear specific rules to avoid duplicates if re-initialized in the same process
            list(self.prolog.query("retractall(profit_margin(_, _, _))"))
            list(self.prolog.query("retractall(growth_rate(_, _, _, _))"))
            list(self.prolog.query("retractall(op_income_growth(_, _, _, _))"))
        except:
            pass # Pyswip can be temperamental with setup queries

        rules = [
            # Base rules (handle the new 3-argument schema: revenue(Company, Year, Amount))
            
            # Profit margin calculation: profit_margin(Company, Year, Margin)
            "profit_margin(Company, Year, Margin) :- (revenue(Company, Year, Rev) ; operating_income(Company, Year, Rev)), (net_income(Company, Year, NI); operating_income(Company, Year, NI)), Rev > 0, Margin is (NI / Rev) * 100",
            
            # Revenue/Income comparison (same year)
            "higher_metric(Comp1, Comp2, Year, MetricType) :- call(MetricType, Comp1, Year, V1), call(MetricType, Comp2, Year, V2), V1 > V2",
            
            # Growth rate calculation: growth_rate(Company, YearOld, YearNew, Rate)
            "growth_rate(Company, YearOld, YearNew, Rate) :- revenue(Company, YearOld, RevOld), revenue(Company, YearNew, RevNew), YearNew > YearOld, RevOld > 0, Rate is ((RevNew - RevOld) / RevOld) * 100",
            
            # Operating income growth
            "op_income_growth(Company, YearOld, YearNew, Rate) :- operating_income(Company, YearOld, VOld), operating_income(Company, YearNew, VNew), YearNew > YearOld, VOld > 0, Rate is ((VNew - VOld) / VOld) * 100",

            # Total Allocated Cost calculation
            "total_allocated_cost(Company, Year, Total) :- findall(Amt, allocated_cost(Company, Year, _, Amt), Amts), sum_list(Amts, Total)",

            # VaR Comparison
            "var_diff(Company, Year1, Year2, Diff) :- value_at_risk(Company, Year1, V1), value_at_risk(Company, Year2, V2), Diff is V1 - V2",

            # First time condition met
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
    
    # Growth Rate Test Scenario (New Schema)
    print("--- Testing Growth Rate (New Schema) ---")
    kb.add_fact("revenue(apple, 2022, 100000)") # 100 billion
    kb.add_fact("revenue(apple, 2023, 120000)") # 120 billion
    
    results, proof = kb.query("growth_rate(apple, 2022, 2023, G)")
    if results:
        growth = results[0]['G']
        print(f"Revenue Growth Rate: {growth}% ✅")
    else:
        print("Growth calculation failed. ❌")
