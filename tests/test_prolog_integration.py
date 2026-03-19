import pytest
from prolog_rag_project.core.prolog_kb import PrologKnowledgeBase

@pytest.fixture
def kb():
    """Returns a fresh PrologKnowledgeBase instance."""
    return PrologKnowledgeBase()

def test_fact_addition(kb):
    """Verifies that facts can be asserted and retrieved from Prolog."""
    kb.add_fact("revenue(apple, 2023, 383000)")
    results, _ = kb.query("revenue(apple, 2023, X)")
    assert len(results) == 1
    assert results[0]['X'] == 383000

def test_query_execution(kb):
    """Verifies arbitrary query execution."""
    kb.add_fact("company(nvidia)")
    results, _ = kb.query("company(nvidia)")
    assert len(results) == 1

def test_profit_margin_calculation(kb):
    """Verifies the profit margin reasoning rule."""
    # Data: Revenue 100, Net Income 20 -> Margin 20%
    # Note: The rule in prolog_kb.py uses (revenue or operating_income) and (net_income or operating_income)
    kb.add_fact("revenue(test_co, 2024, 100)")
    kb.add_fact("net_income(test_co, 2024, 20)")
    
    # query: profit_margin(Company, Year, Margin)
    results, _ = kb.query("profit_margin(test_co, 2024, M)")
    assert len(results) == 1
    assert float(results[0]['M']) == 20.0

def test_growth_rate_calculation(kb):
    """Verifies temporal reasoning for growth rates."""
    # Data: 2022=100, 2023=150 -> Growth 50%
    kb.add_fact("revenue(test_co, 2022, 100)")
    kb.add_fact("revenue(test_co, 2023, 150)")
    
    # growth_rate(Company, YearOld, YearNew, Rate)
    results, _ = kb.query("growth_rate(test_co, 2022, 2023, G)")
    assert len(results) == 1
    assert float(results[0]['G']) == 50.0
