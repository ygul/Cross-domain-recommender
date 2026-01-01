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
chatbot/
├── chat_orchestrator.py       # Main orchestration logic (chat + elicitation + retrieval)
├── chat_ui_cli.py             # Thin CLI interface for humans
├── 3_evaluate_rag.py          # Automated LLM-as-a-Judge evaluation script
├── clarify_gate.py            # LLM-driven preference elicitation (Clarify Gate)
├── vector_store.py            # Vector database wrapper
├── query_embedder.py          # Embedding and query enhancement
├── llm_adapter.py             # LLM provider abstraction (OpenAI / HuggingFace)
├── preference_elicitor_llm.py # Logic for generating questions and handling dialogue
├── rag_evaluatie_metrics.xlsx # Output file containing evaluation scores
└── config.ini                 # Configuration file
design/
└── (Diagrams and visual documentation)
logs/
└── elicitation_log.csv        # Logs of user interactions