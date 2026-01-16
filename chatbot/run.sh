#!/bin/bash
set -e

# Zorg dat we in de juiste map zitten
cd "$(dirname "$0")"

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
"$PY" -m pip install chromadb sentence-transformers huggingface_hub python-dotenv openai pandas openpyxl matplotlib scikit-learn seaborn> /dev/null

# 4. Run commands
# Let op: ik heb 'chatbot/' weggehaald voor de bestandsnamen

if [ "$1" == "ingest" ]; then
    echo "📥 Start Data Ingestie..."
    "$PY" -u 1_importeer_data.py
    "$PY" -u 2_check_en_visualiseer.py

elif [ "$1" == "eval" ]; then
    echo "⚖️  Start Evaluatie (LLM Judge)..."
    # AANGEPAST: Direct het bestand aanroepen, geen submap
    "$PY" -u Judge_module.py

elif [ "$1" == "chat" ]; then
    echo "💬 Start Chat Interface..."
    # AANGEPAST: Direct het bestand aanroepen
    if command -v winpty &> /dev/null; then
        winpty "$PY" -u chat_orchestrator.py
    else
        "$PY" -u chat_orchestrator.py
    fi

else
    echo "Gebruik: ./run.sh [ingest|eval|chat]"
fi