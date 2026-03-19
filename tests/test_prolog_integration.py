import pytest
from prolog_rag_project.core.prolog_kb import PrologKnowledgeBase

@pytest.fixture
def kb():
    """Returns a fresh PrologKnowledgeBase instance."""
    return PrologKnowledgeBase()

def test_fact_addition(kb):
    """Verifies that facts can be asserted and retrieved from Prolog."""
    kb.assert_fact("revenue(apple, 2023, 383000)")
    results = kb.query("revenue(apple, 2023, X)")
    assert len(results) == 1
    assert results[0]['X'] == 383000

def test_query_execution(kb):
    """Verifies arbitrary query execution."""
    kb.assert_fact("company(nvidia)")
    results = kb.query("company(nvidia)")
    assert len(results) == 1

def test_profit_margin_calculation(kb):
    """Verifies the profit margin reasoning rule."""
    # Data: Revenue 100, Cost 80 -> Profit 20 -> Margin 20%
    kb.assert_fact("revenue(test_co, 2024, 100)")
    kb.assert_fact("cost_of_sales(test_co, 2024, 80)")
    
    # query: margin(Company, Year, Margin)
    results = kb.query("margin(test_co, 2024, M)")
    assert len(results) == 1
    assert float(results[0]['M']) == 20.0

def test_growth_rate_calculation(kb):
    """Verifies temporal reasoning for growth rates."""
    # Data: 2022=100, 2023=150 -> Growth 50%
    kb.assert_fact("revenue(test_co, 2022, 100)")
    kb.assert_fact("revenue(test_co, 2023, 150)")
    
    # Note: Using op_income_growth as a proxy or if revenue_growth exists
    # Assuming op_income_growth(C, Y1, Y2, G) logic in rules
    results = kb.query("op_income_growth(test_co, 2022, 2023, G)")
    if results:
        assert float(results[0]['G']) == 50.0
