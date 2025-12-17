from pathlib import Path
import io
import configparser
import requests
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# -------------------------------------------------
# Config & paden
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.ini"

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

DB_ROOT = (BASE_DIR / config["BESTANDEN"]["database_mapnaam"]).resolve()
DATA_DIR = DB_ROOT / "data"

print(f"Chroma DB root: {DB_ROOT}")
print(f"Data map: {DATA_DIR}")

# -------------------------------------------------
# Download Google Sheet
# -------------------------------------------------
sheet_id = config["BESTANDEN"]["google_sheet_id"]
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"

print("⬇Downloaden Google Sheet...")
response = requests.get(url, timeout=60)
if response.status_code != 200:
    raise RuntimeError("Google Sheet niet bereikbaar (check sharing settings)")

df = pd.read_excel(io.BytesIO(response.content))
print("Excel succesvol gedownload")

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
# Chroma setup (NOOIT nieuwe map maken)
# -------------------------------------------------
client = chromadb.PersistentClient(path=str(DB_ROOT))

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

collection_name = config["BESTANDEN"]["collectie_naam"]
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=ef
)

# -------------------------------------------------
# Upsert data
# -------------------------------------------------
documents = df[text_col].tolist()
ids = df[id_col].tolist()

df[meta_cols] = df[meta_cols].fillna("")
metadatas = df[meta_cols].to_dict(orient="records")

print("Vectoriseren & opslaan...")
collection.upsert(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"Klaar. Collectie '{collection_name}' bevat nu {collection.count()} items")
