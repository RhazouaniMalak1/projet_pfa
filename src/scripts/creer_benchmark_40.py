import os
from pathlib import Path
import pandas as pd
import requests

HEADERS = {
    "User-Agent": "PFA_Etudiant_Benchmark - Version 1.0 (projet.etudiant@univ.fr)"
}


def trouver_dossier_data():
    """Détecte dynamiquement l'emplacement du dossier data."""
    chemins = [
        Path(__file__).resolve().parent / "data",
        Path.cwd() / "data",
        Path.cwd() / "src" / "scripts" / "data",
    ]
    for c in chemins:
        if (c / "openfoodfacts_clean.csv").exists():
            return c
    # Par défaut dans src/scripts/data
    d = Path(__file__).resolve().parent / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def recuperer_image_off(code):
    """
    Interroge l'API Open Food Facts pour récupérer l'URL de l'image.
    Priorité 1 : Le tableau nutritionnel (gros plan)
    Priorité 2 : Les ingrédients
    Priorité 3 : La face avant
    """
    url = f"https://world.openfoodfacts.org/api/v0/product/{code}.json"
    try:
        res = requests.get(url, headers=HEADERS, timeout=8)
        if res.status_code == 200:
            prod = res.json().get("product", {})
            url_nutrition = prod.get("image_nutrition_url")
            url_ingredients = prod.get("image_ingredients_url")
            url_front = prod.get("image_front_url") or prod.get("image_url")
            
            # Priorité au tableau nutritionnel
            url_choisie = url_nutrition or url_ingredients or url_front
            type_img = "nutrition" if url_nutrition else ("ingredients" if url_ingredients else "face")
            return url_choisie, type_img
    except Exception:
        pass
    return None, None


def preparer_benchmark():
    dossier_data = trouver_dossier_data()
    chemin_csv = dossier_data / "openfoodfacts_clean.csv"

    if not chemin_csv.exists():
        print(f"❌ Impossible de trouver {chemin_csv}")
        return

    df = pd.read_csv(chemin_csv)

    # Filtrer les candidats ayant des informations nutritionnelles de référence
    candidats = df[
        (df["ingredients_text"].notna()) & 
        (df["ingredients_text"] != "Non renseigné") &
        (df["energy_kcal"].notna())
    ].copy()

    # Mélanger pour varier les produits
    candidats = candidats.sample(frac=1.0, random_state=42).reset_index(drop=True)

    dossier_images = dossier_data / "images_benchmark_40"
    dossier_images.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Recherche et téléchargement de 40 vraies étiquettes dans '{dossier_images}'...")

    lignes_selectionnees = []
    telechargees = 0

    for _, row in candidats.iterrows():
        if telechargees >= 40:
            break

        code = str(row["code"]).strip()
        nom = str(row["product_name"])[:28]

        # Récupération de la meilleure URL (priorité nutrition)
        img_url = row.get("image_nutrition_url") if ("image_nutrition_url" in row and pd.notna(row.get("image_nutrition_url"))) else None
        type_img = "nutrition" if img_url else None

        if not img_url:
            img_url, type_img = recuperer_image_off(code)

        if not img_url:
            continue

        chemin_img = dossier_images / f"{code}.jpg"

        # Téléchargement de l'image
        try:
            res_img = requests.get(img_url, headers=HEADERS, timeout=10)
            if res_img.status_code == 200 and len(res_img.content) > 5000:  # Image valide (>5Ko)
                with open(chemin_img, "wb") as f:
                    f.write(res_img.content)
                
                telechargees += 1
                row_dict = row.to_dict()
                row_dict["image_url"] = img_url
                row_dict["type_image"] = type_img
                row_dict["fichier_image"] = f"{code}.jpg"
                lignes_selectionnees.append(row_dict)

                print(f"  ✅ [{telechargees}/40] Téléchargée ({type_img}) : {nom}")
            else:
                continue
        except Exception:
            continue

    # Sauvegarde du fichier de vérité terrain avec exactement 40 produits
    df_benchmark = pd.DataFrame(lignes_selectionnees)
    chemin_benchmark = dossier_data / "ground_truth_40.csv"
    df_benchmark.to_csv(chemin_benchmark, index=False, encoding="utf-8")

    print("\n🎉 BENCHMARK CRÉÉ AVEC SUCCÈS !")
    print(f"📁 Fichier étalon (Ground Truth) : {chemin_benchmark}")
    print(f"🖼️ Images disponibles (40/40)   : {dossier_images}/")


if __name__ == "__main__":
    preparer_benchmark()