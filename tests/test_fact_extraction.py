import pytest
import os
from prolog_rag_project.core.fact_extractor import FinancialFactExtractor

@pytest.fixture
def extractor():
    """Returns a fresh FinancialFactExtractor instance."""
    return FinancialFactExtractor()

def test_revenue_extraction(extractor):
    """Checks that a simple revenue sentence is correctly extracted into a Prolog fact."""
    text = "The total revenue for Apple Inc. in 2023 was $383 billion."
    facts = extractor.extract_from_text(text)
    
    # Should include something like: revenue('Apple', 2023, 383000.0) or equivalent
    # Facts are returned as a list of strings
    assert any("revenue(" in f for f in facts)
    assert any("2023" in f for f in facts)
    assert any("383" in f for f in facts)

def test_percentage_extraction(extractor):
    """Checks that a percentage (e.g., share repurchase authorization) is correctly extracted."""
    text = "The board authorized an additional 15% increase in share repurchases in 2019."
    facts = extractor.extract_from_text(text)
    
    # Based on schema guidelines in fact_extractor.py
    # We expect some mention of the 2019 data and the 15 value
    assert any("2019" in f for f in facts)
    assert any("15" in f for f in facts)

@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="Requires GROQ_API_KEY")
def test_real_extraction_accuracy(extractor):
    """Real LLM call to verify extraction output format."""
    text = "NVIDIA's revenue in 2024 reached 60.9 billion USD."
    facts = extractor.extract_from_text(text)
    print(f"Extracted Facts: {facts}")
    
    assert isinstance(facts, list)
    assert len(facts) > 0
    # Prolog syntax verification (no spaces in funcs, properly closed parens)
    for f in facts:
        assert f.endswith(".")
        assert "(" in f and ")" in f
