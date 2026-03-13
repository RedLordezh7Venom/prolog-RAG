# Prolog-RAG: Financial Question Answering with Explainable Logical Reasoning

Prolog-RAG is a hybrid system that combines semantic search (Vector RAG) with formal logical reasoning (Prolog) to answer complex financial queries that require numerical calculations, multi-hop reasoning, and temporal analysis.

## Core Features

- **Exact Numerical Answers:** Performs precise calculations for margins, growth rates, etc.
- **Explainable Reasoning:** Every answer includes a full proof trace of the logical steps taken.
- **Multi-hop Reasoning:** Connects facts across multiple documents.
- **Constraint Satisfaction:** Filters results based on complex logical conditions.

## Project Structure

```
prolog-rag-financial/
├── prolog_rag_project/       # Main package
│   ├── core/                 # Core reasoning and extraction logic
│   ├── baselines/            # Comparison systems (Naive RAG, GraphRAG)
│   └── utils/                # Evaluation and visualization
├── notebooks/                # Demos and analysis
├── tests/                    # Unit and integration tests
├── data/                     # Evaluation datasets and results
└── docs/                     # Detailed documentation and analysis
```

## Getting Started

### Prerequisites

- Python 3.11+
- SWI-Prolog 9.x
- Local GPU recommended for LLM fact extraction

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install SWI-Prolog on your system.

## License

MIT License
