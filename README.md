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

## Project Structure

```text
Cross-domain-recommender/
├── data/
│   ├── raw/                        # Original source files
│   │   └── complete_dataset.xlsx   # The initial dataset (books/series/films)
│   └── chromadb/                   # Vector database storage
│       ├── chroma.sqlite3          # SQLite component of Chroma
│       └── bin/                    # Database binaries and libraries
├── src/                            # Source code (the logic)
│   ├── chatbot/                    # Chatbot component (artifact)
│   │   ├── core/                   # Internal chatbot logic
│   │   │   ├── chat_orchestrator.py
│   │   │   ├── query_embedder.py
│   │   │   ├── vector_store.py
│   │   │   ├── clarify_gate.py     # Logic for clarification (includes smoke tests)
│   │   │   ├── llm_adapter.py
│   │   │   └── results_formatter.py
│   │   └── ui/                     # User interface components
│   │       └── chat_ui_cli.py      # Command line interface
│   ├── judge/                      # Evaluation and judging logic
│   │   ├── judge_module.py         
│   │   ├── metrics.py              # Performance and quality metrics
│   │   └── report_generator.py
│   └── utils/                      # Shared utility functions
├── output/                         # All generated files
│   ├── charts/                     # Visualizations and plots
│   └── logs/                       # Interaction and system logs
├── notebooks/                      
│   └── llm_book_processing.ipynb   # Initial prototyping and exploration of data enrichment
├── 1_enrich_dataset.py             # STEP 1: Process and expand raw data
├── 2_import_and_visualize.py       # STEP 2: Load data into Chroma and show stats
├── 3_run_chatbot.py                # STEP 3: Start the conversational interface
├── 4_run_judge.py                  # STEP 4: Run evaluations and generate reports
├── config.ini                      # Global configuration settings
├── requirements.txt                # Project dependencies
├── README.md                       
├── test_scenarios.txt              # Test scenarios for Judge
└── .gitignore                      