# ChromaDB – Cross-Domain Recommender

Deze module bouwt en beheert een **lokale Chroma vector database** op basis van
data die automatisch wordt gedownload uit een **Google Sheet**.
De database wordt gebruikt voor experimenten met **semantic search** en
**cross-domain recommendations**.

De setup is zo ingericht dat de code **direct vanuit de Git-repository**
kan worden uitgevoerd, zonder handmatige stappen.

## Hoe dit project wordt uitgevoerd

Alle acties verlopen via het **run-script** in de root van de repository:

```bash
./run.sh
```

## Architectuur

```text
experiment/
└── chromadb/
    ├── chroma.sqlite3        # Chroma metadata (automatisch beheerd)
    ├── data/                # Vector data (automatisch beheerd door Chroma)
    └── scripts/
        ├── 1_importeer_data.py
        ├── 2_check_en_visualiseer.py
        └── config.ini
run.sh
requirements.txt
```