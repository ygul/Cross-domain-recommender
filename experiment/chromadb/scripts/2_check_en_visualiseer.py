## Libraries ########################################################################################################
#
import chromadb
import configparser
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

#
## Config.ini-bestand ##############################################################################################
#
print("Inlezen config.ini bestand")

config = configparser.ConfigParser()
config.read('config.ini')

# Haal de mapnaam op uit de config
db_folder = config['BESTANDEN']['database_mapnaam']

#
## Aanmaken systeem-mappen ##########################################################################################
#
print("Controle of database-map bestaat")

if not os.path.exists(db_folder):
    os.makedirs(db_folder)
#
if not os.path.exists(db_folder):
    print(f"FOUT: Kan database map '{db_folder}' niet vinden.")
    exit()

#
## Verbinden met database ###########################################################################################
#
print("Verbinden met database")

client = chromadb.PersistentClient(path=db_folder)
collection = client.get_collection(name="cross_domain_recommender")

count = collection.count()					# Tellen aantal items in database ter controle
print(f"Aantal items in database: {count}")

if count == 0:
    print("Database is leeg.")
    exit()

#
## Informatie voor plot ###########################################################################################
#
print("Informatie verzamelen voor controle-plots")

result = collection.get(include=['embeddings', 'metadatas'])
embeddings = result['embeddings']
metadatas = result['metadatas']

#
## Visualisaties ##################################################################################################
#
print("Visualisaties maken")

## PCA (breng embedding dimensies terug naar 2)
pca = PCA(n_components=2)
vis_dims = pca.fit_transform(embeddings)

x = [v[0] for v in vis_dims]
y = [v[1] for v in vis_dims]

## Opbouwen plot
plt.figure(figsize=(10, 8))

kleur_veld = 'Simplified genre'
labels = [m.get(kleur_veld, 'Onbekend') for m in metadatas]

unique_labels = list(set(labels))
colors = [unique_labels.index(l) for l in labels]
    
scatter = plt.scatter(x, y, c=colors, cmap='tab10', alpha=0.7)
handles, _ = scatter.legend_elements()
plt.legend(handles, unique_labels, title=kleur_veld, loc="best", bbox_to_anchor=(1, 1))
	
plt.title(f"Cross-domain recommender - PCA for {count} vectors")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

#
## Opslaan visualisaties ##########################################################################################
#
print("Plots opslaan")

output_file = "visualisatie.png"
plt.savefig(output_file)
print(f"Afbeelding opgeslagen als: {output_file}")
plt.show()