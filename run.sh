#!/bin/bash
set -e

# Zorg dat we in de juiste map zitten
cd "$(dirname "$0")"

# ----------------------------------------
# Laad .env als die bestaat (lokaal)
# ----------------------------------------
if [ -f ".env" ]; then
  echo "🔐 .env geladen"
  export $(grep -v '^\s*#' .env | grep -v '^\s*$' | xargs)
else
  echo "⚠️ Geen .env bestand gevonden (optioneel)."
fi

# OPENAI_API_KEY aanwezigheidstest
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "⚠️ OPENAI_API_KEY is niet gezet. Zet dit in .env of als environment variable."
fi

VENV_DIR=".venv"

# 1. Check of .venv bestaat, zo niet: maak aan
if [ ! -d "$VENV_DIR" ]; then
    echo "🔨 Virtual environment aanmaken..."
    python3 -m venv "$VENV_DIR" || python -m venv "$VENV_DIR"
fi

# 2. Activeer venv
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

PY="python"

# 3. Installeer dependencies (stil)
echo "📦 Dependencies checken..."
"$PY" -m pip install --upgrade pip > /dev/null
"$PY" -m pip install chromadb sentence-transformers huggingface_hub python-dotenv openai pandas openpyxl matplotlib scikit-learn seaborn > /dev/null

# 4. Run commands
# Let op: ik heb 'chatbot/' weggehaald voor de bestandsnamen

if [ "$1" == "ingest" ]; then
    echo "📥 Start Data Ingestie..."
    "$PY" -u ChromaDualModelImporter.py
    "$PY" -u ChromaEmbeddingVisualizer.py

elif [ "$1" == "eval" ]; then
    echo "⚖️  Start Evaluatie (LLM Judge)..."
    # AANGEPAST: Direct het bestand aanroepen, geen submap
    "$PY" -u chatbot/Judge_module.py

elif [ "$1" == "chat" ]; then
    echo "💬 Start Chat Interface..."
    # AANGEPAST: Direct het bestand aanroepen
    if command -v winpty &> /dev/null; then
        winpty "$PY" -u chatbot/chat_ui_cli.py
    else
        "$PY" -u chatbot/chat_ui_cli.py
    fi

else
    echo "Gebruik: ./run.sh [ingest|eval|chat]"
fi
