from pathlib import Path
import configparser
import chromadb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

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

# -------------------------------------------------
# Database connect
# -------------------------------------------------
client = chromadb.PersistentClient(path=str(DB_ROOT))

# Haal alle collecties op
collections = client.list_collections()

if not collections:
    raise RuntimeError("Geen collecties gevonden in de database")

print(f"✔ Aantal collecties: {len(collections)}")

# -------------------------------------------------
# Visualisatie per collectie
# -------------------------------------------------
for collection in collections:
    count = collection.count()
    
    if count == 0:
        print(f"⚠ Collectie '{collection.name}' is leeg, overslaan...")
        continue
    
    embedding_model = collection.metadata.get("embedding_model", "UNKNOWN")
    print(f"\n📊 Visualiseren: '{collection.name}' ({embedding_model}) - {count} vectors")
    
    result = collection.get(include=["embeddings", "metadatas"])
    embeddings = np.array(result["embeddings"])
    metadatas = result["metadatas"]
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    
    labels = [m.get("Simplified genre", "Onbekend") for m in metadatas]
    unique = sorted(set(labels))
    colors = [unique.index(l) for l in labels]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="tab10", alpha=0.7)
    handles, _ = scatter.legend_elements()
    plt.legend(handles, unique, title="Genre", bbox_to_anchor=(1, 1))
    plt.title(f"Chroma DB – PCA visualisatie\nCollectie: {collection.name}\nEmbedding model: {embedding_model}")
    plt.grid(True)
    plt.tight_layout()
    
    filename = f"visualisatie_{collection.name}.png"
    plt.savefig(filename)
    print(f"✔ {filename} opgeslagen")
    plt.close()

print("\n✔ Alle visualisaties aangemaakt")