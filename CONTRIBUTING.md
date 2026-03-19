# Contributing to Prolog-RAG

Thank you for your interest in Prolog-RAG! We welcome contributions to improve the accuracy, reasoning, and scalability of our financial query answering system.

## How to Contribute

1.  **Report Bugs**: Open an issue if you find a bug in fact extraction or reasoning logic.
2.  **Suggest Rules**: Propose new financial reasoning rules (Prolog predicates) in `prolog_kb.py`.
3.  **Improve Benchmarks**: Add new datasets or questions to our `test_questions.json`.
4.  **Submit PRs**: 
    - Fork the repository.
    - Create a feature branch.
    - Run the arena benchmark (`python arena.py`) to ensure no regression.
    - Submit the pull request for review.

## Coding Style
- Follow PEP 8 for Python code.
- Ensure all Prolog rules are well-documented.

## Running Tests
Run all benchmarks using the unified runner:
```bash
python benchmarks/runner.py
```
