from pyswip import Prolog

def test_financial_fact():
    try:
        prolog = Prolog()
        
        # 1. Add a financial fact (scaled to millions to avoid long overflow in pyswip)
        # 394000 represents 394,000 million (394 Billion)
        fact = "revenue(apple, 394000)"
        prolog.assertz(fact)
        print(f"Added fact: {fact} (Value represents millions)")
        
        # 2. Query it back
        query = "revenue(apple, Amount)"
        results = list(prolog.query(query))
        
        if results:
            amount = results[0]['Amount']
            print(f"Query successful! Found Apple revenue (in millions): {amount:,} ✅")
            print(f"Actual Value: ${amount * 1_000_000:,}")
        else:
            print("Query failed: Fact not found in Knowledge Base. ❌")
            
    except Exception as e:
        print(f"Error during Prolog operation: {e}")

if __name__ == "__main__":
    test_financial_fact()
