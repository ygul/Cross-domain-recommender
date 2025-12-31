# Cross-domain Recommender

A semantic search system that recommends books, TV series, and movies based on user queries.  
Uses embeddings and vector similarity search to find relevant items across multiple domains.

---

## Features

- **Semantic Search**  
  Uses sentence embeddings (via an LLM) to understand user intent beyond keyword matching.

- **Multi-Domain Support**  
  Cross-domain search across books, TV series, and movies.  
  Optional item-type filtering is available but disabled by default.

- **Flexible Filtering**  
  Optional filtering by item type(s) with OR logic (feature-flagged in the CLI).

- **LLM Integration**  
  Supports LLM-based query enhancement (OpenAI, extensible by design).

- **LLM-driven Preference Elicitation**  
  The system autonomously decides whether:
  - no follow-up question is needed,
  - one follow-up question is needed,
  - or two follow-up questions are needed (maximum),
  based on the sufficiency of the user’s input.

- **Explainable Recommendations**  
  Each recommendation includes a cosine distance score and an interpretation  
  (Strong / Good / Moderate / Weak match).

- **Vector Database**  
  Persistent storage using ChromaDB.

- **Logging for Evaluation**  
  All preference elicitation sessions are logged for analysis and reproducibility.

- **CLI Interface**  
  Thin command-line interface for interaction.

---

## Architecture Notes

All conversational logic, including preference elicitation and the decision
whether to ask follow-up questions, is centralized in the `ChatOrchestrator`.

The CLI layer is intentionally kept thin and only handles input/output.
This design enables loose coupling and allows alternative frontends
(e.g. an LLM-as-a-judge module) to fully simulate the CLI by calling
`ChatOrchestrator.chat(...)` directly.

---

## Project Structure

```
chatbot/
├── chat_orchestrator.py       # Main orchestration logic (chat + elicitation + retrieval)
├── chat_ui_cli.py             # Thin CLI interface (no chat logic)
├── clarify_gate.py            # LLM-driven preference elicitation (Clarify Gate)
├── vector_store.py            # Vector database wrapper
├── query_embedder.py          # Embedding and query enhancement
├── llm_adapter.py             # LLM provider abstraction
├── results_formatter.py       # Result formatting incl. cosine distance
├── elicitation_logger.py      # CSV / JSONL logging
├── test_elicitation.py        # Reproducible elicitation tests
└── config.ini                 # Configuration file
logs/
└── elicitation_log.csv / elicitation_log.jsonl
```

---

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key
- Existing ChromaDB collection

### Installation

1. **Create virtual environment**
   ```bash
   python -m venv .chatbot
   source .chatbot/bin/activate  # On Windows: .chatbot\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   Create a `.env` file in the project root with your API credentials:
   ```
   OPENAI_API_KEY=your_key_here
   ```

4. **Configure database**
   Ensure `chatbot/config.ini` is in sync with your database paths and collection name.

---

## Usage

### CLI Interface

```bash
cd chatbot
python chat_ui_cli.py
```

The CLI will:
- prompt the user for a natural-language description,
- optionally ask follow-up questions to clarify preferences,
- return ranked recommendations with explained similarity scores.

---

### Programmatic Usage

```python
from chat_orchestrator import ChatOrchestrator

orchestrator = ChatOrchestrator(llm_provider="openai", max_items=3)

result = orchestrator.run_once("I want a good sci-fi book")
print(result)
```

---

## Preference Elicitation

Before retrieval, the system performs **preference elicitation**.

An **elicitation session** consists of:
1. The initial user query (seed)
2. Zero or more follow-up questions asked by the LLM
3. User answers to those questions
4. A final semantic query (`final_query`) used for vector search

The LLM determines autonomously when sufficient information has been collected.

---

## Semantic Similarity & Distance

- The vector store uses **cosine distance** over sentence embeddings.
- Lower cosine distance indicates a stronger semantic match.

Distance interpretation:
- `0.00 – 0.20` → **Strong match**
- `0.20 – 0.40` → **Good match**
- `0.40 – 0.60` → **Moderate match**
- `> 0.60` → **Weak match**

Cosine distance values are explicitly shown and interpreted in the chatbot output.

---

## Testing & Evaluation

### Reproducible Elicitation Tests

```bash
python chatbot/test_elicitation.py
```

The test script demonstrates:
- 0 follow-up questions
- 1 follow-up question
- 2 follow-up questions

For each scenario, it outputs:
- asked questions and answers,
- the final retrieval query,
- chatbot recommendations including cosine distance,
- and logs the session to CSV and JSONL.

---

## Logging

Each elicitation session is logged with:
- user seed input
- follow-up questions and answers
- final semantic query
- item type filter (`ALL` when no filtering is applied)

Logs are written to:
- `logs/elicitation_log.csv`
- `logs/elicitation_log.jsonl`

---

## Configuration

Edit `chatbot/config.ini` to customize:
- database location
- collection name
- other ChromaDB-related settings

---

## Requirements

See `requirements.txt` for the full dependency list:
- chromadb
- sentence-transformers
- openai

---