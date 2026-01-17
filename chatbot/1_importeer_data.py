from pathlib import Path
import configparser
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import shutil
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# -------------------------------------------------
# Config & paden
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[0]

CONFIG_PATH = BASE_DIR / "config.ini"
excel_path = REPO_ROOT / "data" / "complete dataset.xlsx"

config = configparser.ConfigParser()

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"config.ini niet gevonden: {CONFIG_PATH}")

config.read(CONFIG_PATH)

if "BESTANDEN" not in config:
    raise KeyError(f"[BESTANDEN] ontbreekt in {CONFIG_PATH}. Gevonden secties: {config.sections()}")

DB_ROOT = (BASE_DIR / config["BESTANDEN"]["database_mapnaam"]).resolve()
DATA_DIR = DB_ROOT / "data"

print(f"Chroma DB root: {DB_ROOT}")
print(f"Data map: {DATA_DIR}")

print(f"📄 Excel inlezen vanaf: {excel_path}")
if not excel_path.exists():
    raise FileNotFoundError(
        f"Excel bestand niet gevonden: {excel_path}\n"
        f"Zorg dat het bestand bestaat op: {REPO_ROOT / 'data' / 'complete dataset.xlsx'}"
    )

df = pd.read_excel(excel_path)
print("✅ Excel succesvol ingelezen")

# -------------------------------------------------
# Opschonen data
# -------------------------------------------------
id_col = config["DATA"]["kolom_id"]
text_col = config["DATA"]["kolom_tekst"]
meta_cols = [c.strip() for c in config["DATA"]["kolommen_metadata"].split(",")]

df[id_col] = df[id_col].astype(str)

df = df.dropna(subset=[text_col])
df = df[df[text_col].apply(lambda x: isinstance(x, str) and x.strip() != "")]

print(f"Aantal geldige rijen: {len(df)}")

# -------------------------------------------------
# Chroma setup - Verwijder oude database
# -------------------------------------------------
if DB_ROOT.exists():
    print(f"🗑️ Verwijderen oude database: {DB_ROOT}")
    shutil.rmtree(DB_ROOT)

print(f"📦 Aanmaken nieuwe database: {DB_ROOT}")
client = chromadb.PersistentClient(path=str(DB_ROOT))

# Lees beide embedding modellen uit config
model_1 = config["AI"]["model_embedding"]
model_alternative = config["AI"]["model_embedding_alternative"]

print(f"🔤 Model 1: {model_1}")
print(f"🔤 Model alternative: {model_alternative}")

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_1
)

collection_name = config["BESTANDEN"]["collectie_naam"]
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=ef,
    metadata={
        "hnsw:space": "cosine",
        "embedding_model": ef.model_name
    }
)

# Voor een tweede model
ef2 = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_alternative
)

collection_name_model2 = config["BESTANDEN"]["collectie_naam_model2"]
collection_model2 = client.get_or_create_collection(
    name=collection_name_model2,
    embedding_function=ef2,
    metadata={
        "hnsw:space": "cosine",
        "embedding_model": ef2.model_name
    }
)

# -------------------------------------------------
# Upsert data
# -------------------------------------------------
documents = df[text_col].tolist()
ids = df[id_col].tolist()

df[meta_cols] = df[meta_cols].fillna("")
metadatas = df[meta_cols].to_dict(orient="records")

print("Vectoriseren & opslaan - model 1...")
collection.upsert(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"Klaar. Collectie '{collection_name}' bevat nu {collection.count()} items")

# -------------------------------------------------
# Upsert data for second model
# -------------------------------------------------
print("Vectoriseren & opslaan - model 2...")
collection_model2.upsert(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"Klaar. Collectie '{collection_name_model2}' bevat nu {collection_model2.count()} items")

# -------------------------------------------------
# Test: ophalen van de modelnamen uit de vector store
# -------------------------------------------------
print("\nBeschikbare embedding modellen in de collectie:")
collections = client.list_collections()
for col in collections:
    embedding_model = col.metadata.get("embedding_model", "UNKNOWN")
    print(f"- Collectie '{col.name}': model '{embedding_model}'")
