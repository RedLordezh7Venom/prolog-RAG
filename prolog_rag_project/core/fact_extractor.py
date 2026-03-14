import re
import logging

logger = logging.getLogger(__name__)

class FactExtractor:
    """
    Extracts structured facts (Prolog predicates) from financial text.
    Uses a hybrid approach of Regex and (eventually) LLM.
    """
    def __init__(self):
        self.patterns = {
            'revenue': re.compile(r'revenue of \$?([\d,\.]+)\s*(million|billion|trillion)?', re.IGNORECASE),
            'net_income': re.compile(r'net income of \$?([\d,\.]+)\s*(million|billion|trillion)?', re.IGNORECASE),
            'year': re.compile(r'\b(19|20)\d{2}\b')
        }

    def _normalize_amount(self, amount_str, multiplier_str):
        """
        Converts currency strings to integers scaled to MILLIONS.
        This prevents overflow issues in the pyswip bridge (expected 'long').
        Example: '394.3 billion' -> 394300
        """
        amount = float(amount_str.replace(',', ''))
        if multiplier_str:
            mult = multiplier_str.lower()
            if 'trillion' in mult:
                amount *= 1_000_000 # 1T = 1,000,000M
            elif 'billion' in mult:
                amount *= 1_000 # 1B = 1,000M
            elif 'million' in mult:
                amount *= 1 # 1M = 1M
        return int(amount)

    def extract_from_text(self, text, doc_id=None):
        """
        Extracts basic facts using regex.
        Returns a list of formatted Prolog fact strings: 'predicate(arg1, arg2)'
        """
        facts = []
        
        # Extract year
        year_match = self.patterns['year'].search(text)
        year = year_match.group(0) if year_match else "unknown_year"
        
        # Use provided doc_id, or fallback to doc_year
        # Replace non-alphanumeric chars in doc_id to make it a valid Prolog atom
        safe_doc_id = str(doc_id or f"doc_{year}").replace(' ', '_').replace('-', '_').replace('.', '_').replace('/', '_').lower()

        # Extract revenue
        for match in self.patterns['revenue'].finditer(text):
            amount = self._normalize_amount(match.group(1), match.group(2))
            facts.append(f"revenue('{safe_doc_id}', {amount})")

        # Extract net income
        for match in self.patterns['net_income'].finditer(text):
            amount = self._normalize_amount(match.group(1), match.group(2))
            facts.append(f"net_income('{safe_doc_id}', {amount})")

        return facts

    def extract_llm_facts(self, text):
        """To be implemented in later phases using Llama 3.1."""
        # Placeholder for complex entity/relation extraction
        return []

    def extract_all(self, text, doc_id=None):
        """Combines regex and LLM extraction."""
        facts = self.extract_from_text(text, doc_id)
        facts.extend(self.extract_llm_facts(text))
        return facts

if __name__ == "__main__":
    extractor = FactExtractor()
    test_text = "Apple reported revenue of $394.3 billion"
    print(f"Testing text: '{test_text}'")
    
    facts = extractor.extract_from_text(test_text, "apple_2023")
    print("\nExtracted facts:")
    for fact in facts:
        print(f" - {fact}")
