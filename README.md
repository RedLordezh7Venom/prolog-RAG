<p align="center">
  <img src="https://img.shields.io/badge/Prolog--RAG-Financial--AI-2e7d32?style=for-the-badge&logo=prolog&logoColor=white" height="40">
</p>

<h1 align="center">Prolog-RAG: Formal Logic for Financial Reasoning</h1>

<p align="center">
  <a href="https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white">
    <img src="https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://img.shields.io/badge/Logic-SWI--Prolog-red?style=flat-square&logo=prolog&logoColor=white">
    <img src="https://img.shields.io/badge/Logic-SWI--Prolog-red?style=flat-square&logo=prolog&logoColor=white" alt="SWI-Prolog">
  </a>
  <a href="https://img.shields.io/badge/LLM-Llama--3.1--8B-orange?style=flat-square&logo=groq&logoColor=white">
    <img src="https://img.shields.io/badge/LLM-Llama--3.1--8B-orange?style=flat-square&logo=groq&logoColor=white" alt="LLM">
  </a>
  <a href="https://img.shields.io/badge/license-MIT-green?style=flat-square">
    <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
  </a>
</p>

<p align="center">
  <b>A hybrid RAG system bridging the gap between semantic retrieval and symbolic precision.</b>
</p>

---

## 🛑 The Problem with Traditional RAG

Standard Vector-based RAG architectures consistently fail in the financial domain because:

*   **Numerical Hallucinations**: LLMs struggle with multi-step arithmetic, leading to incorrect calculations for growth rates, total costs, and margins.
*   **Logical Multi-Hop Gaps**: Information scattered across fragmented documents (e.g., across 2012–2014 SEC filings) results in "lost-in-the-middle" reasoning errors.
*   **Mathematical Imprecision**: Simple semantic search cannot handle complex constraints or temporal comparative logic (e.g., "Find the year with the lowest margin").
*   **Opaque Reasoning**: Answers are generated as "black boxes," providing no verifiable audit trail—a critical failure for regulatory compliance.

## 🚀 The Solution: Prolog-RAG

Prolog-RAG solves these issues by offloading **Reasoning** from the LLM to a **Symbolic Logic Engine**.

*   **Symbolic Fact Extraction**: Uses LLMs to turn natural language context into structured Prolog predicates (`revenue(aal, 2023, 500)`).
*   **Deterministic Reasoning**: Executes complex financial rules (growth, margin, threshold analysis) through a symbolic Prolog backend.
*   **100% Numerical Accuracy**: Performs exact mathematical calculations, eliminating the "rounding" errors inherent in LLM-based synthesis.
*   **Verifiable Proof Traces**: Every answer comes with a step-by-step logic trace, showing exactly which financial facts and reasoning rules were used to derive the result.

---

## ⚡ Quick Start

Get your Prolog-RAG environment running in minutes:

```bash
# 1. Clone the repository
git clone https://github.com/RedLordezh7Venom/prolog-RAG.git && cd prolog-RAG

# 2. Install dependencies (requires SWI-Prolog 9.x+)
uv pip install -e .

# 3. Run the interactive demo
uv run python demo_app.py
```

---

## 📊 Benchmark Results

Evaluated on our **Grounded Financial QA Suite** (10 high-stakes financial reasoning questions):

| System | Avg Accuracy Score | Proof Trace Availability | Best For... |
| :--- | :---: | :---: | :--- |
| **🟢 Prolog-RAG** | **4.6 / 5** | **100% (10/10)** | **Logic, Arithmetic, Auditing** |
| **🔵 Contextual RAG** | 4.8 / 5 | 0% | General Semantic Lookups |
| **🔴 Naive RAG** | 3.2 / 5 | 0% | Basic FAQ retrieval |
| **🟣 Graph RAG** | 0.7 / 5 | 0% | Complex entity mapping |

---

## 🗺️ Project Architecture

```text
prolog-rag/
├── prolog_rag_project/
│   ├── core/           # Hybrid pipeline, Query Router, NL-to-Prolog translator
│   ├── baselines/      # Naive, Graph, CRAG, and Contextual RAG implementations
│   └── utils/          # Auto-Evaluator, Reporting, & Visualization tools
├── benchmarks/         # Data generators for NIAH, HotpotQA, and FRAMES
├── docs/               # Comparative analysis and PRD documentation
└── assets/             # Performance charts and visualizations
```

## 📜 Explainability Trace Example

```text
Query: "What was the growth in Technical Solutions operating income from 2017 to 2018?"
Trace:
  -> fact: operating_income('Technical Solutions', 2017, 21.0)
  -> fact: operating_income('Technical Solutions', 2018, 32.0)
  -> rule: op_income_growth(Company, 2017, 2018, Growth)
  -> calc: (32.0 - 21.0) / 21.0 * 100 = 52.38%
Answer: "The operating income grew by 52.4%."
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
