#!/bin/bash
set -e

cd "$(dirname "$0")"

VENV_DIR=".venv"

# 1. Check of .venv bestaat, zo niet: maak aan
if [ ! -d "$VENV_DIR" ]; then
    echo "🔨 Virtual environment aanmaken..."
    python3 -m venv "$VENV_DIR"
fi

# 2. Activeer venv (optioneel, maar handig voor lokale shell)
if [ -f "$VENV_DIR/bin/activate" ]; then
    # Linux/macOS
    source "$VENV_DIR/bin/activate"
else
    # Windows Git Bash fallback
    source "$VENV_DIR/Scripts/activate"
fi

PY="$VENV_DIR/bin/python"
if [ ! -f "$PY" ]; then
  # Windows Git Bash fallback
  PY="$VENV_DIR/Scripts/python"
fi

# 3. Installeer dependencies (altijd via dezelfde interpreter)
echo "📦 Dependencies checken/installeren..."
"$PY" -m pip install --upgrade pip
"$PY" -m pip install \
  chromadb \
  sentence-transformers \
  huggingface_hub \
  python-dotenv \
  openai \
  pandas \
  openpyxl \
  matplotlib \
  scikit-learn

# 4. Run
if [ "$1" == "ingest" ]; then
    echo "📥 Start Data Ingestie..."
    "$PY" 1_importeer_data.py
    "$PY" 2_check_en_visualiseer.py

elif [ "$1" == "eval" ]; then
    echo "⚖️  Start Evaluatie (LLM Judge)..."
    "$PY" 3_evaluate_rag.py

elif [ "$1" == "chat" ]; then
    echo "💬 Start Chat Interface..."
    if command -v winpty &> /dev/null; then
        winpty "$PY" chat_ui_cli.py
    else
        "$PY" chat_ui_cli.py
    fi

else
    echo "Gebruik: ./run.sh [ingest|eval|chat]"
    echo "  ingest : Database bouwen (script 1 & 2)"
    echo "  eval   : RAG Evaluatie draaien (script 3)"
    echo "  chat   : Zelf praten met de bot (chat_ui_cli.py)"
fi
