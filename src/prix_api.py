"""
src/prix_api.py - Gestion hybride des prix : 
1. API Open Prices (relevés officiels)
2. Régression ML avec intervalle [Prix Min, Prix Estimé, Prix Max]
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests

URL_OPEN_PRICES = "https://prices.openfoodfacts.org/api/v1/prices"
HEADERS = {"User-Agent": "NutriScanAI-PFA/1.0 (etudiant.data@univ.ma)"}

_MODELE_PRIX = None


def _charger_regresseur():
    global _MODELE_PRIX
    if _MODELE_PRIX is None:
        chemins = [
            Path("data/modele_prix_regresseur.pkl"),
            Path("src/scripts/data/modele_prix_regresseur.pkl"),
            Path(__file__).resolve().parent / "data" / "modele_prix_regresseur.pkl"
        ]
        for p in chemins:
            if p.exists():
                try:
                    _MODELE_PRIX = joblib.load(p)
                    break
                except Exception:
                    pass
    return _MODELE_PRIX


def estimer_fourchette_prix_ml(produit: dict) -> dict:
    """
    Estime une fourchette de prix [Min, Moyen, Max] en Dirhams (MAD)
    en exploitant la distribution des 150 arbres du Random Forest.
    """
    modele = _charger_regresseur()
    if modele is None:
        return {"prix_min": 4.5, "prix_estime": 7.5, "prix_max": 10.5, "methode": "Valeur moyenne par défaut"}

    champs = produit.get("champs") or {}
    off = produit.get("off") or {}

    def _extraire(cle_champ, cle_off, defaut=0.0):
        for v in [champs.get(cle_champ), off.get(cle_off), produit.get(cle_champ)]:
            if v not in [None, "inconnu", "None", "", np.nan]:
                try:
                    return float(str(v).replace(",", ".").replace("g", "").replace("kcal", "").strip())
                except (ValueError, TypeError):
                    pass
        return defaut

    cal = _extraire("energie_kcal", "energy_kcal", 180.0)
    gras = _extraire("graisses", "fat_100g", 6.0)
    sucre = _extraire("sucres", "sugars_100g", 8.0)
    prot = _extraire("proteines", "proteins_100g", 4.0)

    cols = ["energie_kcal", "fat_100g", "sugars_100g", "proteins_100g"]
    df_vect = pd.DataFrame([[cal, gras, sucre, prot]], columns=cols)

    try:
        # Prédictions de chacun des arbres du Random Forest
        predictions_arbres = [arbre.predict(df_vect.values)[0] for arbre in modele.estimators_]
        
        prix_estime = round(float(np.mean(predictions_arbres)), 2)
        prix_min = round(float(np.percentile(predictions_arbres, 15)), 2)
        prix_max = round(float(np.percentile(predictions_arbres, 85)), 2)

        return {
            "prix_min": max(1.5, prix_min),
            "prix_estime": prix_estime,
            "prix_max": max(prix_estime, prix_max),
            "methode": "Random Forest Regressor (Distribution empirique)"
        }
    except Exception:
        return {"prix_min": 4.5, "prix_estime": 7.5, "prix_max": 10.5, "methode": "Secours standard"}


def recuperer_prix_api(produit_ou_code) -> dict:
    code_barres = ""
    produit_dict = {}

    if isinstance(produit_ou_code, dict):
        produit_dict = produit_ou_code
        code_barres = str(produit_ou_code.get("code_barres") or produit_ou_code.get("code") or "").strip()
    else:
        code_barres = str(produit_ou_code).strip()
        produit_dict = {"code": code_barres}

    # 1. Tenter l'API Open Prices officielle
    if code_barres and code_barres.isdigit():
        try:
            params = {"product_code": code_barres, "size": 1}
            rep = requests.get(URL_OPEN_PRICES, params=params, headers=HEADERS, timeout=4)
            if rep.status_code == 200:
                items = rep.json().get("items", [])
                if items:
                    dernier = items[0]
                    p = dernier.get("price")
                    devise = dernier.get("currency", "MAD")
                    p_mad = round(float(p) * 10.8 if devise == "EUR" else float(p), 2)
                    magasin = dernier.get("location_osm_name") or "Magasin partenaire"
                    date_releve = str(dernier.get("date", ""))[:10] or "Récent"

                    return {
                        "disponible": True,
                        "prix_estime": p_mad,
                        "prix_min": round(p_mad * 0.92, 2),
                        "prix_max": round(p_mad * 1.08, 2),
                        "devise": "MAD",
                        "magasin": magasin,
                        "date": date_releve,
                        "source": "Open Prices API",
                        "methode": "Relevé officiel en rayon"
                    }
        except Exception:
            pass

    # 2. Fallback vers le modèle Machine Learning (Fourchette [Min, Max])
    fourchette = estimer_fourchette_prix_ml(produit_dict)
    return {
        "disponible": True,
        "prix_estime": fourchette["prix_estime"],
        "prix_min": fourchette["prix_min"],
        "prix_max": fourchette["prix_max"],
        "devise": "MAD",
        "magasin": "Marché marocain (Estimation)",
        "date": "Calculé par IA",
        "source": "Modèle Random Forest",
        "methode": fourchette["methode"]
    }


obtenir_prix_produit = recuperer_prix_api