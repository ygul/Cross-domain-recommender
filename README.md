# Cross-domain Recommender

A semantic search system that recommends books, TV series, and movies based on user queries.
Uses embeddings and vector similarity search to find relevant items across multiple domains.

---

## Features

- **Semantic Search**
  Uses sentence embeddings (via an LLM) to understand user intent beyond keyword matching.

- **Multi-Domain Support**
  Cross-domain search across books, TV series, and movies.

- **LLM-driven Preference Elicitation**
  The system autonomously decides whether to ask follow-up questions (0, 1, or 2 max) based on the sufficiency of the user’s input.

- **Automated Evaluation (LLM-as-a-Judge)**
  Includes a fully automated testing framework where a "Simulated User" interacts with the chatbot, and a "Judge LLM" scores the performance and improvement.

- **Explainable Recommendations**
  Each recommendation includes a cosine distance score and an interpretation (Strong / Good / Moderate / Weak match).

- **Vector Database**
  Persistent storage using ChromaDB.

---

## Architecture Notes

All conversational logic is centralized in the `ChatOrchestrator`.

The system supports two modes of interaction:
1. **Human Mode:** Via the CLI, where a real user interacts with the bot.
2. **Evaluation Mode:** The CLI is bypassed. A `Simulated User` (LLM) generates answers based on a hidden intent, and a `Judge` (LLM) evaluates the quality of the recommendations.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
```

### Configure environment
Create a `.env` file in the project root with your API credentials:
```text
OPENAI_API_KEY=your_key_here
```

### Configure database
Ensure `config.ini` is in sync with your database paths and collection name.

---

## Usage

The project includes a `run.sh` script to automate setup and execution. This script will automatically create a virtual environment and install dependencies if they are missing.

### 1. Data Ingestion
Initializes the database, imports data, and generates embeddings.
```bash
./run.sh ingest
```

### 2. Chat Interface
Starts the interactive CLI chatbot (Human Mode).
```bash
./run.sh chat
```

### 3. Evaluation
Runs the automated LLM-as-a-Judge evaluation (Evaluation Mode).
```bash
./run.sh eval
```

---
## Project Structure

```text
Cross-domain-recommender/
├── chatbot/                        # Core source code
│   ├── chat_orchestrator.py        # Central logic
│   ├── chat_ui_cli.py              # Command line interface
│   ├── clarify_gate.py             # Logic for clarification
│   ├── Definities_rag_evaluation_final.txt
│   ├── Judge_module.py             # Evaluation module
│   ├── llm_adapter.py              # LLM communication
│   ├── metrics.py                  # Performance metrics
│   ├── query_embedder.py           # Embedding logic
│   ├── report_generator.py         # Reporting tools
│   ├── results_formatter.py        # Output formatting
│   ├── simulated_user.py           # Simulated user for evaluation
│   ├── test_scenarios.txt          # Scenarios for testing
│   ├── vector_store.py             # Database interactions
│   └── working_adapter.py          # Adapter implementation
├── data/
│   ├── chromadb/                   # Vector database storage
│   ├── output/                     # Generated files
│   └── raw/                        # Original source files
├── design/                         # Design documents and diagrams
├── experiment/                     # Console outputs from experiments
├── notebook/                       # Jupyter notebooks
│   └── generate_enriched_descriptions.ipynb
├── ChromaDualModelImporter.py      # Script to import data
├── ChromaEmbeddingVisualizer.py    # Script to check and visualize data
├── config.ini                      # Configuration
├── LICENSE                         # License
├── requirements.txt                # Dependencies
├── run.sh                          # Shell script to run the project
└── README.md
```