#!/usr/bin/env bash
set -euo pipefail

# Root = map waar dit run.sh bestand staat (dus experiment/chromadb)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV="${ROOT}/chromadb"
REQ_FILE="${ROOT}/requirements.txt"

IMPORT_SCRIPT="${ROOT}/scripts/1_importeer_data.py"
PLOT_SCRIPT="${ROOT}/scripts/2_check_en_visualiseer.py"

usage() {
  echo "Gebruik: ./run.sh [import|plot|all]"
}

# 1) venv
if [[ ! -d "$VENV" ]]; then
  echo "➕ Virtuele omgeving aanmaken: chromadb"
  python3 -m venv "$VENV"
fi

# 2) activate
# shellcheck disable=SC1090
source "${VENV}/bin/activate"

# 3) deps
python -m pip install --upgrade pip
pip install -r "$REQ_FILE"

# 4) WSL deps (optioneel)
if command -v apt >/dev/null 2>&1; then
  sudo apt install -y python3-tk ca-certificates >/dev/null
fi

case "${1:-all}" in
  import)
    python "$IMPORT_SCRIPT"
    ;;
  plot)
    python "$PLOT_SCRIPT"
    ;;
  all)
    python "$IMPORT_SCRIPT"
    python "$PLOT_SCRIPT"
    ;;
  *)
    usage
    exit 1
    ;;
esac