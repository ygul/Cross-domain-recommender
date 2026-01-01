# Cross-domain Recommender

A semantic search system that recommends books, TV series, and movies based on user queries.  
Uses embeddings and vector similarity search to find relevant items across multiple domains.

---

## Features

- **Semantic Search** Uses sentence embeddings (via an LLM) to understand user intent beyond keyword matching.

- **Multi-Domain Support** Cross-domain search across books, TV series, and movies.  
  Optional item-type filtering is available but disabled by default.

- **Flexible Filtering** Optional filtering by item type(s) with OR logic (feature-flagged in the CLI).

- **LLM Integration** Supports LLM-based query enhancement (OpenAI, extensible by design).

- **LLM-driven Preference Elicitation** The system autonomously decides whether:
  - no follow-up question is needed,
  - one follow-up question is needed,
  - or two follow-up questions are needed (maximum),
  based on the sufficiency of the user’s input.

- **Automated Evaluation (LLM-as-a-Judge)**
  Includes a testing framework where a "Simulated User" interacts with the chatbot and a "Judge" scores the performance (USR, Precision, and Improvement).

- **Explainable Recommendations** Each recommendation includes a cosine distance score and an interpretation  
  (Strong / Good / Moderate / Weak match).

- **Vector Database** Persistent storage using ChromaDB.

- **Logging for Evaluation** All preference elicitation sessions are logged for analysis and reproducibility.

- **CLI Interface** Thin command-line interface for interaction.

---

## Architecture Notes

All conversational logic, including preference elicitation and the decision
whether to ask follow-up questions, is centralized in the `ChatOrchestrator`.

The CLI layer is intentionally kept thin and only handles input/output.
This design enables loose coupling and allows alternative frontends
(e.g. the LLM-as-a-judge module) to fully simulate the CLI by calling
`ChatOrchestrator.chat(...)` directly.

---

## Project Structure