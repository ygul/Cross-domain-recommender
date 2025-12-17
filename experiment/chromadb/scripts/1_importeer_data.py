## Libraries ########################################################################################################
#

import pandas as pd
import chromadb
import configparser
import os
import io
import requests
from chromadb.utils import embedding_functions

#
## Config.ini-bestand ##############################################################################################
#
print("Inlezen config.ini bestand")

config = configparser.ConfigParser()
config.read('config.ini')
sheet_id = config['BESTANDEN']['google_sheet_id']
db_folder = config['BESTANDEN']['database_mapnaam']

#
## Aanmaken systeem-mappen ##########################################################################################
#
print("Controle of database-map bestaat anders aanmaken")

if not os.path.exists(db_folder):
    os.makedirs(db_folder)
#
## Downloaden excel met data ########################################################################################
#
print("Start met downloaden excel met data van google drive")

url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
response = requests.get(url)

if response.status_code == 200:
    df = pd.read_excel(io.BytesIO(response.content))
    print("Excel succesvol gedownload!")
else:
    print(f"FOUT: Kon bestand niet downloaden. Status code: {response.status_code}")
    exit()

#
## Opschonen informatie in excel ####################################################################################
#
print("Start met opschonen informatie in excel")

id_col = config['DATA']['kolom_id']
df[id_col] = df[id_col].astype(str)
print(f"kolom gebruikt als unieke identifier: {id_col}")

txt_col = config['DATA']['kolom_tekst']
meta_cols = config['DATA']['kolommen_metadata'].split(',')

print(f"Aantal rijen gevonden: {len(df)}")

df = df.dropna(subset=[txt_col]) 				# Verwijder rijen waar de tekst-kolom LEEG/NaN is
df = df[df[txt_col].apply(lambda x: isinstance(x, str))] 	# Verwijder rijen waar de tekst-kolom geen tekst is

print(f"Aantal rijen na verwijderen van lege/foute regels: {len(df)}")

#
## Lokale database setup #############################################################################################
#
print("Start met database setup")

client = chromadb.PersistentClient(path=db_folder)

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")

Projectnaam = config['BESTANDEN']['collectie_naam']
collection = client.get_or_create_collection(name=Projectnaam, embedding_function=ef)

#
## Data in database zetten ###########################################################################################
#
print("Start met data vectoriseren")

documents = df[txt_col].tolist()
ids = df[id_col].tolist()

df[meta_cols] = df[meta_cols].fillna('') # Als er NaN in de metadata staat, maak er lege string van 
metadatas = df[meta_cols].to_dict(orient='records')

batch_size = 100
for i in range(0, len(documents), batch_size):
    end = min(i + batch_size, len(documents))
    print(f"Verwerken rij {i} tot {end}")
    
    collection.upsert(
        documents=documents[i:end],
        metadatas=metadatas[i:end],
        ids=ids[i:end]
    )

#
## Vervolgstappen ####################################################################################################
#
print(f"Database gevuld en beschikbaar in '{db_folder}'.")