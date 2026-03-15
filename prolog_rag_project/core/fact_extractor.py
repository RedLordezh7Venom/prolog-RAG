import re
import json
import logging
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class FinancialFactExtractor:
    """
    Extracts structured facts (Prolog predicates) from financial text.
    Uses a hybrid approach of Regex and (eventually) LLM.
    """
    def __init__(self):
        self.llm = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        
        # Prolog schema definition for the prompt
        self.schema = """
        Use the following Prolog predicate formats:
        1. revenue(Company, Year, AmountInMillions).
        2. net_income(Company, Year, AmountInMillions).
        3. profit_margin_target(Company, Year, Percentage).
        4. growth_target(Company, Year, Percentage).
        
        Guidelines:
        - Company: Convert to a lowercase, single_word atom (e.g., 'apple_inc' -> apple).
        - Year: Extract a 4-digit integer. If missing, use 9999.
        - AmountInMillions: ALWAYS convert dollar amounts to an integer representing MILLIONS (e.g., "$394 billion" -> 394000).
        - Percentage: Extract as a float (e.g., "24.6%" -> 24.6).
        """

    def extract_llm_facts(self, text, doc_id=None):
        """
        Uses Llama 3.3 to extract highly structured facts mapping to Prolog predicates.
        """
        prompt = f"""
        Extract financial facts from the following text and format them exactly as a JSON list of Prolog predicates.
        
        {self.schema}
        
        Text to analyze:
        {text[:2000]}
        
        Respond ONLY with a JSON object containing a "facts" array. 
        Example format: {{"facts": ["revenue(apple, 2023, 394300)", "net_income(apple, 2023, 96900)"]}}
        """
        
        try:
            completion = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            response_json = json.loads(completion.choices[0].message.content)
            facts = response_json.get("facts", [])
            
            # Basic validation to ensure they look like predicates
            valid_facts = [f for f in facts if "(" in f and ")" in f]
            return valid_facts
        except Exception as e:
            logger.error(f"LLM Extraction failed: {e}")
            return []

    def extract_all(self, text, doc_id=None):
        """Main entry point. We now rely primarily on LLM extraction for robustness."""
        return self.extract_llm_facts(text, doc_id)

if __name__ == "__main__":
    extractor = FinancialFactExtractor()
    
    # Test LLM Extraction
    test_text = "In the fiscal year 2023, Apple Inc. reported a total revenue of $394.3 billion. The company's net income for the period stood at $96.9 billion, representing a significant portion of its earnings."
    print(f"Testing text: '{test_text}'")
    
    print("\nExtracting via LLM...")
    facts = extractor.extract_all(test_text, "doc_1")
    
    print("\nExtracted Prolog Facts:")
    for fact in facts:
        print(f" - {fact}")
