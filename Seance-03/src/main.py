#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/

# Sources des données : production de M. Forriez, 2016-2023

# main.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Création des dossiers data et img s'ils n'existent pas
os.makedirs("data", exist_ok=True)
os.makedirs("img", exist_ok=True)


# Partie 1 : Résultats élections

csv_elections = "data/resultats-elections-presidentielles-2022-1er-tour.csv"

# Lecture du fichier CSV
with open(csv_elections, "r", encoding="utf-8") as f:
    df_elec = pd.read_csv(f)

# Sélection des colonnes quantitatives
quant_cols = df_elec.select_dtypes(include=np.number).columns

# Calcul des statistiques
stats = pd.DataFrame(index=quant_cols)

stats['moyenne'] = df_elec[quant_cols].mean().round(2)
stats['mediane'] = df_elec[quant_cols].median().round(2)
stats['mode'] = df_elec[quant_cols].mode().iloc[0].round(2)
stats['ecart_type'] = df_elec[quant_cols].std().round(2)
stats['ecart_abs_moyenne'] = (df_elec[quant_cols] - df_elec[quant_cols].mean()).abs().mean().round(2)
stats['etendue'] = (df_elec[quant_cols].max() - df_elec[quant_cols].min()).round(2)

# Distance interquartile et interdécile
stats['iqr'] = (df_elec[quant_cols].quantile(0.75) - df_elec[quant_cols].quantile(0.25)).round(2)
stats['id'] = (df_elec[quant_cols].quantile(0.9) - df_elec[quant_cols].quantile(0.1)).round(2)

print("\nStatistiques élections (quantitative) :\n")
print(stats)

# Boîtes à moustache
for col in quant_cols:
    plt.figure()
    df_elec.boxplot(column=col)
    plt.title(f"Boîte à moustache : {col}")
    plt.savefig(f"img/{col}_boxplot.png")
    plt.close()

# Export CSV et Excel
stats.to_csv("data/statistiques_elections.csv", index=True)
stats.to_excel("data/statistiques_elections.xlsx", index=True)

# ------------------------------
# Partie 2 : Island index
# ------------------------------
csv_islands = "data/island-index.csv"

with open(csv_islands, "r", encoding="utf-8") as f:
    df_islands = pd.read_csv(f)

# Catégorisation des surfaces
bins = [0, 10, 25, 50, 100, 2500, 5000, 10000, np.inf]
labels = [
    "0-10", "10-25", "25-50", "50-100",
    "100-2500", "2500-5000", "5000-10000", ">=10000"
]

df_islands['Surface_cat'] = pd.cut(df_islands['Surface (km²)'], bins=bins, labels=labels, right=True)

# Comptage par catégorie
surface_counts = df_islands['Surface_cat'].value_counts().sort_index()
print("\nCatégorisation des surfaces des îles :\n")
print(surface_counts)

# Export CSV et Excel
surface_counts.to_csv("data/surface_categorisee.csv", index=True)
surface_counts.to_excel("data/surface_categorisee.xlsx", index=True)


