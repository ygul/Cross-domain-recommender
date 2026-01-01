#!/bin/bash

# Naam van de virtual environment map
VENV_DIR="venv"

# 1. Check of venv bestaat, zo niet: maak aan
if [ ! -d "$VENV_DIR" ]; then
    echo "🔨 Virtual environment aanmaken..."
    python3 -m venv $VENV_DIR
fi

# 2. Activeer venv
# (Op Windows Git Bash werkt 'source' meestal ook, anders ./venv/Scripts/activate)
if [ -f "$VENV_DIR/bin/activate" ]; then
    source $VENV_DIR/bin/activate
else
    source $VENV_DIR/Scripts/activate
fi

# 3. Installeer dependencies (NU MET OPENAI, PANDAS en OPENPYXL)
echo "📦 Dependencies checken/installeren..."
pip install --upgrade pip
pip install chromadb sentence-transformers huggingface_hub python-dotenv openai pandas openpyxl

# 4. Argument afhandeling
if [ "$1" == "ingest" ]; then
    echo "📥 Start Data Ingestie..."
    python 1_importeer_data.py
    python 2_check_en_visualiseer.py

elif [ "$1" == "eval" ]; then
    echo "⚖️  Start Evaluatie (LLM Judge)..."
    python 3_evaluate_rag.py

elif [ "$1" == "chat" ]; then
    echo "💬 Start Chat Interface..."
    # Gebruik winpty als het beschikbaar is (voorkomt vastlopen in Git Bash op Windows)
    if command -v winpty &> /dev/null; then
        winpty python chat_ui_cli.py
    else
        python chat_ui_cli.py
    fi

else
    echo "Gebruik: ./run.sh [ingest|eval|chat]"
    echo "  ingest : Database bouwen (script 1 & 2)"
    echo "  eval   : RAG Evaluatie draaien (script 3)"
    echo "  chat   : Zelf praten met de bot (chat_ui_cli.py)"
fi