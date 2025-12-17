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
collection_name = config["BESTANDEN"]["collectie_naam"]

collection = client.get_collection(name=collection_name)
count = collection.count()

if count == 0:
    raise RuntimeError("Collectie is leeg")

print(f"✔ Aantal vectors: {count}")

# -------------------------------------------------
# Visualisatie
# -------------------------------------------------
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
plt.title("Chroma DB – PCA visualisatie")
plt.grid(True)
plt.tight_layout()

plt.savefig("visualisatie.png")
print("visualisatie.png opgeslagen")

try:
    plt.show()
except Exception:
    pass