import os
import pandas as pd
import requests

HEADERS = {
    "User-Agent": "PFA_Etudiant_Benchmark - Version 1.0 (projet.etudiant@univ.fr)"
}


def recuperer_image_off(code):
    """Interroge l'API Open Food Facts pour trouver l'URL de l'image du produit."""
    url = f"https://world.openfoodfacts.org/api/v0/product/{code}.json"
    try:
        res = requests.get(url, headers=HEADERS, timeout=8)
        if res.status_code == 200:
            prod = res.json().get("product", {})
            # On cherche l'image des ingrédients ou la face avant
            return (
                prod.get("image_ingredients_url")
                or prod.get("image_url")
                or prod.get("image_front_url")
            )
    except Exception:
        pass
    return None


def preparer_benchmark():
    chemin_csv = "data/openfoodfacts_clean.csv"
    if not os.path.exists(chemin_csv):
        print(f"❌ Impossible de trouver {chemin_csv}")
        return

    df = pd.read_csv(chemin_csv)

    # 1. Sélectionner les produits ayant des ingrédients exploitables
    df_candidats = df[
        (df["ingredients_text"].notna())
        & (df["ingredients_text"] != "Non renseigné")
    ].copy()

    # Prendre 40 produits variés
    benchmark_40 = df_candidats.sample(
        n=min(40, len(df_candidats)), random_state=42
    ).copy()

    # 2. Créer le dossier pour les photos
    dossier_images = "data/images_benchmark_40"
    os.makedirs(dossier_images, exist_ok=True)

    print(
        f"🔍 Recherche et téléchargement des 40 images dans '{dossier_images}'..."
    )

    urls_images = []
    telechargees = 0

    for idx, row in benchmark_40.iterrows():
        code = str(row["code"])
        nom = str(row["product_name"])[:25]

        # Si l'URL existe déjà dans le CSV on l'utilise, sinon on interroge l'API
        img_url = (
            row.get("image_url")
            if "image_url" in row and pd.notna(row.get("image_url"))
            else None
        )
        if not img_url:
            img_url = recuperer_image_off(code)

        urls_images.append(img_url)

        if img_url:
            chemin_img = os.path.join(dossier_images, f"{code}.jpg")
            if not os.path.exists(chemin_img):
                try:
                    img_data = requests.get(
                        img_url, headers=HEADERS, timeout=10
                    ).content
                    with open(chemin_img, "wb") as f:
                        f.write(img_data)
                    telechargees += 1
                    print(f"  ✅ [{telechargees}/40] Image téléchargée : {nom}")
                except Exception as e:
                    print(f"  ⚠️ Échec téléchargement pour {nom} : {e}")
        else:
            print(f"  ❌ Pas d'image trouvée pour {nom} ({code})")

    # 3. Mettre à jour et enregistrer le CSV avec les URLs réelles
    benchmark_40["image_url"] = urls_images
    chemin_benchmark = "data/ground_truth_40.csv"
    benchmark_40.to_csv(chemin_benchmark, index=False, encoding="utf-8")

    print("\n🎉 TERMINÉ AVEC SUCCÈS !")
    print(f"📁 Fichier étalon enregistré : {chemin_benchmark}")
    print(f"🖼️ Images disponibles dans   : {dossier_images}/")


if __name__ == "__main__":
    preparer_benchmark()