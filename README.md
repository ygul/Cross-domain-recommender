# Cross-domain Recommender

A semantic search system that recommends books, TV series, and movies based on user queries. Uses embeddings and vector similarity search to find relevant items across multiple domains.

## Features

- **Semantic Search**: Uses sentence embeddings (via LLM) to understand user intent beyond keyword matching
- **Multi-Domain Support**: Search and filter across books, TV series, and movies
- **Flexible Filtering**: Optional filtering by item type(s) with OR logic
- **LLM Integration**: Supports LLM-based query enhancement (OpenAI, extensible by design)
- **Vector Database**: Persistent storage using Chroma DB
- **CLI Interface**: Interact via command-line interface

## Project Structure

```
chatbot/
├── chat_orchestrator.py      # Main orchestrator class
├── chat_ui_cli.py            # CLI interface
├── vector_store.py           # Vector database wrapper
├── query_embedder.py         # Embedding and query enhancement
├── llm_adapter.py            # LLM provider abstraction
├── results_formatter.py      # Result formatting
└── config.ini                # Configuration file
```

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   - Create a `.env` file in the project root with your API credentials:
   ```
   OPENAI_API_KEY=your_key_here
   ```

4. **Configure database**:
   - Ensure `chatbot/config.ini` is in sync with your database paths and collection name

## Usage

### CLI Interface

```bash
cd chatbot
python chat_ui_cli.py
```

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

## Configuration

Edit `chatbot/config.ini` to customize:
- Database location
- Collection name
- Other settings

## Requirements

See `requirements.txt` for full dependency list:
- chromadb - Vector database
- sentence-transformers - Embeddings
- openai - LLM provider

## License

MIT License