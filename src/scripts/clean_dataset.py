import os
import re
import numpy as np
import pandas as pd
import requests

HEADERS = {
    "User-Agent": "PFA_Etudiant_AnalyseEtiquettes - Version 1.0 (projet.etudiant@univ.fr)"
}


def telecharger_echantillon_off(nb_pages=5):
    """Télécharge des produits populaires depuis Open Food Facts France."""
    print("⏳ Téléchargement des données en cours depuis Open Food Facts...")
    produits_bruts = []

    for page in range(1, nb_pages + 1):
        url = "https://world.openfoodfacts.org/cgi/search.pl"
        params = {
            "action": "process",
            "tagtype_0": "countries",
            "tag_contains_0": "contains",
            "tag_0": "france",
            "sort_by": "unique_scans_n",
            "page_size": 50,
            "page": page,
            "json": "true",
        }

        try:
            response = requests.get(
                url, params=params, headers=HEADERS, timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                nouveaux = data.get("products", [])
                produits_bruts.extend(nouveaux)
                print(
                    f"✅ Page {page}/{nb_pages} téléchargée ({len(nouveaux)} produits récupérés)"
                )
            else:
                print(f"❌ Erreur HTTP {response.status_code} sur la page {page}")
        except Exception as e:
            print(f"❌ Erreur de connexion sur la page {page} : {e}")

    return produits_bruts


def nettoyer_texte_ingredients(texte):
    """Nettoie les balises markdown, html et retours à la ligne."""
    if not isinstance(texte, str) or texte.strip() == "":
        return "Non renseigné"
    # Supprime les _ (ex: _lait_ -> lait) et les étoiles *
    texte = re.sub(r"[_*]", "", texte)
    # Remplace les retours à la ligne par des espaces
    texte = re.sub(r"[\r\n\t]+", " ", texte)
    # Supprime les espaces multiples
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()


def extraire_et_nettoyer(produits):
    """Filtre les colonnes, récupère les images et nettoie les données."""
    if not produits:
        print("⚠️ Aucun produit n'a été récupéré.")
        return pd.DataFrame()

    print(" Nettoyage des données en cours...")

    lignes = []
    for p in produits:
        nutr = p.get("nutriments", {})

        ligne = {
            "code": p.get("code"),
            "product_name": p.get("product_name"),
            "categories": p.get("categories"),
            "nutriscore_grade": p.get("nutriscore_grade"),
            "ingredients_text": p.get("ingredients_text"),
            # AJOUT INDISPENSABLE : Liens vers les images d'emballage
            "image_url": p.get("image_url") or p.get("image_front_url"),
            "image_ingredients_url": p.get("image_ingredients_url"),
            "energy_kcal": nutr.get("energy-kcal_100g"),
            "fat_100g": nutr.get("fat_100g"),
            "sugars_100g": nutr.get("sugars_100g"),
            "proteins_100g": nutr.get("proteins_100g"),
            "salt_100g": nutr.get("salt_100g"),
            # Traçabilité (Consigne S3)
            "source": "Open Food Facts API",
        }
        lignes.append(ligne)

    df = pd.DataFrame(lignes)

    # 1. Supprimer les lignes où le code ou le nom manque
    df = df.dropna(subset=["code", "product_name"])
    df["code"] = df["code"].astype(str).str.strip()

    # 2. Supprimer les doublons
    df = df.drop_duplicates(subset=["code"])

    # 3. Convertir les nutriments en nombres
    cols_numeriques = [
        "energy_kcal",
        "fat_100g",
        "sugars_100g",
        "proteins_100g",
        "salt_100g",
    ]
    for col in cols_numeriques:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Remplacer les valeurs aberrantes par NaN (sans supprimer le produit entier)
    for col in ["fat_100g", "sugars_100g", "proteins_100g", "salt_100g"]:
        df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan

    df.loc[
        (df["energy_kcal"] < 0) | (df["energy_kcal"] > 950), "energy_kcal"
    ] = np.nan

    # 5. Nettoyer le Nutriscore
    df["nutriscore_grade"] = df["nutriscore_grade"].astype(str).str.lower()
    df.loc[
        ~df["nutriscore_grade"].isin(["a", "b", "c", "d", "e"]),
        "nutriscore_grade",
    ] = "inconnu"

    # 6. Nettoyer le texte
    df["ingredients_text"] = df["ingredients_text"].apply(
        nettoyer_texte_ingredients
    )
    df["categories"] = df["categories"].fillna("Non classé")

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    produits = telecharger_echantillon_off(nb_pages=5)
    df_clean = extraire_et_nettoyer(produits)

    if not df_clean.empty:
        chemin_csv = "data/openfoodfacts_clean.csv"
        df_clean.to_csv(chemin_csv, index=False, encoding="utf-8")

        print("\n🎉 Terminé avec succès !")
        print(f"📁 Fichier enregistré dans : {chemin_csv}")
        print(f"📊 Nombre de produits propres : {len(df_clean)}")
        print("\nAperçu des premières lignes :")
        print(
            df_clean[
                ["product_name", "sugars_100g", "salt_100g", "nutriscore_grade"]
            ].head(5)
        )