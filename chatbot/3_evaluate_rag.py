## Python libraries #########################################################################################################################################
#

import time
import pandas as pd
import re
import configparser
from pathlib import Path
from huggingface_hub import InferenceClient

import chat_orchestrator
from preference_elicitor_llm import PreferenceElicitorLLM
from query_embedder import QueryEmbedder

## Setup ####################################################################################################################################################
#

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.ini"

config = configparser.ConfigParser()

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"config.ini niet gevonden: {CONFIG_PATH}")

config.read(CONFIG_PATH)

if "AI" not in config:
    raise KeyError(f"[AI] ontbreekt in {CONFIG_PATH}. Gevonden secties: {config.sections()}")

# Judge setup (HuggingFace)
JUDGE_MODEL = config["AI"]["model_judge"]
HF_TOKEN = config["AI"]["hf_token"]
client_judge = InferenceClient(token=HF_TOKEN)

# System-to-test setup
print("Initializing Chat Orchestrator")
# 1. Deze regel MOET er staan (anders krijg je NameError: orchestrator not defined)
orchestrator = chat_orchestrator.ChatOrchestrator(llm_provider="openai", max_items=3)

# 2. Deze regel heb je net toegevoegd voor de embedder fix
print("Initializing Independent Embedder")
test_embedder = QueryEmbedder(model_name=config["AI"]["model_embedding"])
