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

## 📖 Table of Contents
*   [🛑 The Problem with Traditional RAG](#-the-problem-with-traditional-rag)
*   [🚀 The Solution: Prolog-RAG](#-the-solution-prolog-rag)
*   [🏗️ System Architecture](#-system-architecture)
*   [🔍 Example: The "Audit Proof" Difference](#-example-the-audit-proof-difference)
*   [📊 Benchmark Results](#-benchmark-results)
*   [🛠️ How It Works](#-how-it-works)
*   [🏗️ Project Structure](#-project-structure)
*   [💻 Tech Stack](#-tech-stack)
*   [🛠️ Installation & Setup](#-installation--setup)
*   [📜 Explainability Trace Example](#-explainability-trace-example)
*   [🤝 Contributing](#-contributing)
*   [📄 License](#-license)

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

## 🏗️ System Architecture

Prolog-RAG uses a hybrid routing mechanism to ensure high precision for structured queries while maintaining the flexibility of semantic search.

```text
       ┌───────────────┐
       │   User Query  │
       └───────┬───────┘
               ▼
       ┌───────────────┐
       │ Query Router  │ (LLM-Based Decision Entry)
       └───────┬───────┘
               │
      ┌────────┴────────┐
      ▼ (Arithmetic)    ▼ (Semantic)
┌───────────────┐   ┌───────────────┐
│  Prolog Path  │   │  Vector Path  │
├───────────────┤   ├───────────────┤
│ Fact Extract  │   │ Chroma Search │
│ Logic Engine  │   │ LLM Synthesis │
└────────┬───────┘   └───────┬───────┘
         │                   │
         └────────┬──────────┘
                  ▼
          ┌───────────────┐
          │  Final Answer │ (With Proof Trace if Prolog)
          └───────────────┘
```

---

## 🔍 Example: The "Audit Proof" Difference

**Query**: *"What was the gross profit margin for the company in 2017?"*

### 🟢 Prolog-RAG (Reasoning Path)
**Answer**: "The gross profit margin for 2017 was 18.91%."  
**Verification Trace**:
```prolog
1. [Extract] revenue(company, 2017, 3314.0).
2. [Extract] cost_of_sales(company, 2017, 2687.0).
3. [Rule]    gross_profit(C, Y, GP) :- revenue(C, Y, R), cost(C, Y, S), GP is R - S.
4. [Rule]    margin(C, Y, M) :- gross_profit(C, Y, G), revenue(C, Y, R), M is (G/R)*100.
5. [Execute] M is ((3314 - 2687) / 3314) * 100 = 18.9197...
```

### 🔴 Traditional RAG (Semantic Path)
**Answer**: "The company reported a strong gross margin in 2017, approximately 19% based on the consolidated statements."  
**Verification Trace**:
> ❌ **None.** Source of the "19%" figure is opaque and subject to LLM rounding/estimation.

---
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

---

## 🛠️ How It Works

1.  **Query Input**: The user provides a natural language financial query.
2.  **Hybrid Routing**: An LLM analyzes the query to determine if it is **Semantic** (FAQ/Summary) or **Arithmetic/Logical** (Calculations/Multi-hop).
3.  **Fact Extraction**: For logical queries, the system retrieves relevant document chunks and extracts structured financial facts (e.g., `revenue(co, 2023, 500)`).
4.  **Symbolic Unification**: The facts are asserted into a **SWI-Prolog** knowledge base alongside domain-specific financial reasoning rules.
5.  **Logic Execution**: The Prolog engine executes a symbolic query to derive the exact numerical answer or logical conclusion.
6.  **Answer Synthesis**: The system generates a natural language answer, appending a **Proof Trace** for full transparency and explainability.

---

## 🏗️ Project Structure

```text
prolog-rag/
├── prolog_rag_project/
│   ├── core/           # Hybrid pipeline, Query Router, NL-to-Prolog translator
│   ├── baselines/      # Naive, Graph, CRAG, and Contextual RAG implementations
│   └── utils/          # Auto-Evaluator, Reporting, & Visualization tools
├── benchmarks/         # Data generators for NIAH, HotpotQA, and FRAMES
├── docs/               # Comparative analysis and PRD documentation
├── assets/             # Performance charts and visualizations
├── demo_app.py         # Streamlit Interactive Demo
└── arena.py            # Unified benchmarking arena
```

---

## 💻 Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Orchestration & Pipeline |
| **Logic Engine** | SWI-Prolog 9.x | Symbolic Reasoning & Arithmetic |
| **LLM (Backbone)** | Llama 3.1 (via Groq) | Fact Extraction & Query Routing |
| **Vector Database**| ChromaDB | Semantic Retrieval & Context Management |
| **Embeddings** | Sentence-Transformers | Vectorizing Financial Documents |
| **Visualization** | Matplotlib | Performance & Benchmark Charting |

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.11 or higher.
*   [SWI-Prolog](https://www.swi-prolog.org/download/stable) installed and added to your system PATH.
*   A **Groq API Key** (Set in `.env`).

### Step-by-Step
1.  **Clone the Repo**:
    ```bash
    git clone https://github.com/RedLordezh7Venom/prolog-RAG.git && cd prolog-RAG
    ```
2.  **Install Dependencies**:
    ```bash
    uv pip install -e .
    ```
3.  **Environment Setup**:
    Create a `.env` file in the root and add your Groq API key:
    ```env
    GROQ_API_KEY=your_key_here
    ```
4.  **Run the Benchmark**:
    ```bash
    uv run python arena.py
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
