import os
import glob
from pathlib import Path

# 1. Désactiver MKLDNN pour éviter le crash CPU avec PaddlePaddle 3.3
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from paddleocr import PaddleOCR

# Import robuste (fonctionne que vous soyez à la racine ou dans src/scripts)
try:
    from parser_s4 import parser_etiquette
except ImportError:
    from src.scripts.parser_s4 import parser_etiquette


def tester_ocr():
    print("⏳ Initialisation du moteur PaddleOCR 3.7 (mode CPU stable)...")
    ocr = PaddleOCR(
        use_textline_orientation=True, 
        lang='fr',
        enable_mkldnn=False
    )

    # Recherche automatique du dossier benchmark
    chemins_possibles = [
        Path(__file__).resolve().parent / "data" / "images_benchmark_40",
        Path.cwd() / "data" / "images_benchmark_40",
        Path.cwd() / "src" / "scripts" / "data" / "images_benchmark_40"
    ]

    dossier_images = None
    for p in chemins_possibles:
        if p.exists() and any(p.iterdir()):
            dossier_images = p
            break

    if not dossier_images:
        print("❌ Dossier 'images_benchmark_40' introuvable !")
        return

    # Récupération des images disponibles
    images = list(dossier_images.glob("*.*"))
    if not images:
        print(f"❌ Aucune image trouvée dans {dossier_images}")
        return

    print(f"\n📁 Dossier trouvé : {dossier_images}")
    print(f"📊 Total d'images disponibles : {len(images)}")
    print("🔍 Test en cours sur les 5 premières images...\n")

    # On teste les 5 premières images pour voir des produits avec vrais tableaux
    for idx, image_path in enumerate(images[:5], start=1):
        nom_fichier = os.path.basename(image_path)
        print("=" * 65)
        print(f"📸 [{idx}/5] Test sur : {nom_fichier}")
        print("=" * 65)

        # Prédiction OCR
        resultats = ocr.predict(str(image_path))
        lignes_texte = []

        print("📝 Lignes détectées :")
        for res in resultats:
            # Format standard PaddleOCR 3.7
            if hasattr(res, "get") and res.get("rec_texts"):
                textes = res["rec_texts"]
                scores = res.get("rec_scores", [1.0] * len(textes))
                for t, s in zip(textes, scores):
                    lignes_texte.append(t)
                    print(f"   [{float(s):.2f}] {t}")
            # Format alternatif json
            elif hasattr(res, "json") and isinstance(res.json, dict):
                donnees_json = res.json.get("res", res.json)
                if "rec_texts" in donnees_json:
                    for t in donnees_json["rec_texts"]:
                        lignes_texte.append(t)
                        print(f"   -> {t}")
            # Format rétrocompatible listes
            elif isinstance(res, (list, tuple)):
                for element in res:
                    if isinstance(element, (list, tuple)) and len(element) >= 2 and isinstance(element[1], (list, tuple)):
                        texte = element[1][0]
                        confiance = element[1][1]
                        lignes_texte.append(texte)
                        print(f"   [{confiance:.2f}] {texte}")

        texte_brut = "\n".join(lignes_texte)

        if not texte_brut.strip():
            print("   ⚠️ Aucun texte lisible trouvé sur cette image.")
            continue

        # Extraction des nutriments via parser_s4
        donnees = parser_etiquette(texte_brut)

        print("\n🎯 RÉSULTAT DU PARSING NUTRITIONNEL :")
        for cle, val in donnees.items():
            print(f"   • {cle:<15} : {val}")
        print("\n")


if __name__ == "__main__":
    tester_ocr()