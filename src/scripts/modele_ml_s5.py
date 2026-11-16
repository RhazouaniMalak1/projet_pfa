import os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# 1. Détection automatique du dataset
chemins_possibles = [
    Path("data/openfoodfacts_clean.csv"),
    Path("src/scripts/data/openfoodfacts_clean.csv"),
    Path(__file__).resolve().parent / "data" / "openfoodfacts_clean.csv"
]

chemin_dataset = next((p for p in chemins_possibles if p.exists()), None)
if not chemin_dataset:
    raise FileNotFoundError("❌ Impossible de trouver 'openfoodfacts_clean.csv'.")

print(f"📂 Chargement du dataset : {chemin_dataset}")
df = pd.read_csv(chemin_dataset)


# 2. Assignation multilingue robuste (Français & Anglais)
def assigner_famille(cat):
    if not isinstance(cat, str) or cat.strip() == "":
        return None
    c = cat.lower()

    # Mots-clés élargis
    if any(m in c for m in ["boisson", "eau", "jus", "soda", "beverage", "drink", "tea", "coffee"]):
        return "Boissons"
    elif any(m in c for m in ["biscuit", "gateau", "gâteau", "snack", "chocolat", "cookie", "candy", "sweet"]):
        return "Biscuits & Snacks"
    elif any(m in c for m in ["lait", "fromage", "yaourt", "dessert", "dairy", "cheese", "yogurt", "milk"]):
        return "Produits Laitiers"
    elif any(m in c for m in ["plat", "conserve", "sauce", "pate", "pâte", "riz", "pain", "meal", "groceries"]):
        return "Épicerie & Plats"
    return None

df["famille"] = df["categories"].apply(assigner_famille)

# 3. Préparation des variables
cols_features = ["energy_kcal", "fat_100g", "sugars_100g", "proteins_100g", "salt_100g"]

# On filtre uniquement sur la cible (famille non nulle)
df_ml = df.dropna(subset=["famille"]).copy()

# On ne garde que les classes avec au moins 5 exemples
comptes = df_ml["famille"].value_counts()
familles_valides = comptes[comptes >= 5].index
df_ml = df_ml[df_ml["famille"].isin(familles_valides)]

print(f"\n📊 Répartition des produits par famille ({len(df_ml)} exemples retenus) :")
for fam, total in df_ml["famille"].value_counts().items():
    print(f"  • {fam:<20} : {total} produits")

X = df_ml[cols_features]
y = df_ml["famille"]

# 4. Partition Train (80%) / Test (20%) stratifiée
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# 5. Création des Pipelines Scikit-Learn (Imputation automatique des NaN)
# Baseline (Hasard le plus fréquent)
baseline_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", DummyClassifier(strategy="most_frequent"))
])
baseline_pipeline.fit(X_train, y_train)

# Modèle Machine Learning (Random Forest)
rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Permet à l'application de prédire même avec des NaN de l'OCR
    ("classifier", RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42))
])
rf_pipeline.fit(X_train, y_train)

# 6. Évaluation comparative
y_pred_base = baseline_pipeline.predict(X_test)
y_pred_rf = rf_pipeline.predict(X_test)

print("\n" + "=" * 65)
print("1. MODÈLE DE BASE (DUMMY BASELINE - SANS IA) :")
print("=" * 65)
print(classification_report(y_test, y_pred_base, zero_division=0))

print("=" * 65)
print("2. MODÈLE MACHINE LEARNING (RANDOM FOREST PIPELINE) :")
print("=" * 65)
print(classification_report(y_test, y_pred_rf, zero_division=0))

# 7. Matrice de Confusion lisible et étiquetée
classes = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred_rf, labels=classes)
df_cm = pd.DataFrame(cm, index=[f"Réel: {c}" for c in classes], columns=[f"Prédit: {c}" for c in classes])

print("=" * 65)
print("📊 MATRICE DE CONFUSION DÉTAILLÉE :")
print("=" * 65)
print(df_cm.to_string())

# 8. Sauvegarde du Pipeline complet pour app.py
chemin_modele = chemin_dataset.parent / "modele_famille.pkl"
joblib.dump(rf_pipeline, chemin_modele)
print(f"\n✅ Pipeline complet (Imputer + Modèle) sauvegardé sous :\n   👉 {chemin_modele}")