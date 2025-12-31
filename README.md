# Cross-domain Recommender

A semantic search system that recommends books, TV series, and movies based on user queries.  
Uses embeddings and vector similarity search to find relevant items across multiple domains.

This project has been extended with **LLM-driven preference elicitation**, **explainable ranking using cosine distance**, and **reproducible evaluation via logging and test scripts**.

---

## Features

- **Semantic Search**  
  Uses sentence embeddings (via an LLM) to understand user intent beyond keyword matching.

- **Multi-Domain Support**  
  Search and filter across books, TV series, and movies.

- **Flexible Filtering**  
  Optional filtering by item type(s) with OR logic.

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
  Interact via a command-line interface.

---

## Project Structure

```
chatbot/
├── chat_orchestrator.py       # Main orchestrator class
├── chat_ui_cli.py             # CLI interface
├── preference_elicitor_llm.py # LLM-driven preference elicitation
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

# Initialize orchestrator
orchestrator = ChatOrchestrator(llm_provider="openai", max_items=3)

# Search without filters
result = orchestrator.run_once("I want a good sci-fi book")

# Search with item type filter(s)
result = orchestrator.run_once(
    "What should I watch?",
    item_types={'movie', 'TV series'}
)

# Search with single type
result = orchestrator.run_once(
    "Recommend a book",
    item_types={'Book'}
)

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

## Vector Store API

### `similarity_search(query_embedding, k=5)`
Performs unfiltered semantic search across all items.

### `filtered_similarity_search(query_embedding, k=5, where=None)`
Performs semantic search with optional Chroma metadata filters.

**Filter Examples**:
```python
# Single item type
where = {"item_type": "Book"}

# Multiple types (OR logic)
where = {"$or": [
    {"item_type": "Book"},
    {"item_type": "movie"}
]}

# Complex filters
where = {"$and": [
    {"item_type": "Book"},
    {"year": {"$gte": 2020}}
]}
```

---

## Testing & Evaluation

### Reproducible Elicitation Tests

A dedicated test script is provided:

```bash
python chatbot/test_elicitation.py
```

This script runs predefined scenarios demonstrating:
- 0 follow-up questions
- 1 follow-up question
- 2 follow-up questions

For each scenario, it outputs:
- the questions asked and answers given,
- the final retrieval query,
- the chatbot recommendations including cosine distance,
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

These logs support evaluation and reproducibility.

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
