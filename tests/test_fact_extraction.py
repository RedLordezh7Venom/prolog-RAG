import pytest
import os
from unittest.mock import MagicMock, patch
from prolog_rag_project.core.fact_extractor import FinancialFactExtractor

@pytest.fixture
def extractor():
    """Returns a fresh FinancialFactExtractor instance."""
    return FinancialFactExtractor()

def test_revenue_extraction(extractor):
    """
    Checks that a simple revenue sentence is correctly extracted into a Prolog fact.
    Mocks the LLM to avoid real API calls in base tests.
    """
    text = "The total revenue for Apple Inc. in 2023 was $383 billion."
    mock_facts = ["revenue(apple, 2023, 383000)."]
    
    with patch.object(extractor, 'extract_llm_facts', return_value=mock_facts):
        facts = extractor.extract_all(text)
        assert any("revenue(apple, 2023, 383000)" in f for f in facts)

def test_percentage_extraction(extractor):
    """Checks that a percentage (e.g., share repurchase authorization) is correctly extracted."""
    text = "The board authorized an additional 15% increase in share repurchases in 2019."
    mock_facts = ["authorized_repurchase(company, 2019, 15.0)."]
    
    with patch.object(extractor, 'extract_llm_facts', return_value=mock_facts):
        facts = extractor.extract_all(text)
        assert any("2019" in f for f in facts)
        assert any("15" in f for f in facts)

@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="Requires GROQ_API_KEY")
def test_real_extraction_accuracy(extractor):
    """Real LLM call to verify extraction output format."""
    text = "NVIDIA's revenue in 2024 reached 60.9 billion USD."
    facts = extractor.extract_all(text)
    
    assert isinstance(facts, list)
    if len(facts) > 0:
        for f in facts:
            assert "(" in f and ")" in f
