import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Charger le dataset propre
df = pd.read_csv("data/openfoodfacts_clean.csv")

print("=== 1. COMPLÉTUDE DES DONNÉES (Valeurs manquantes en %) ===")
missing = (df.isnull().sum() / len(df)) * 100
print(missing[missing > 0])

# Graphique 1 : Taux de champs manquants
plt.figure(figsize=(8, 4))
missing.plot(kind="bar", color="coral")
plt.title("Taux de valeurs manquantes par champ (%)")
plt.ylabel("Pourcentage manquant")
plt.tight_layout()
plt.savefig("data/eda_valeurs_manquantes.png")
plt.close()

# Graphique 2 : Distribution des Sucres et Sel
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df["sugars_100g"].dropna(), kde=True, color="green")
plt.title("Distribution des Sucres (g/100g)")

plt.subplot(1, 2, 2)
sns.histplot(df["salt_100g"].dropna(), kde=True, color="blue")
plt.title("Distribution du Sel (g/100g)")
plt.tight_layout()
plt.savefig("data/eda_distributions.png")
plt.close()

# Graphique 3 : Répartition Nutri-Score officiel
plt.figure(figsize=(6, 4))
df["nutriscore_grade"].value_counts().plot(kind="bar", color="teal")
plt.title("Répartition des Nutri-Scores (OFF)")
plt.tight_layout()
plt.savefig("data/eda_nutriscore.png")
plt.close()

print("\n✅ Graphiques générés avec succès dans le dossier data/ !")