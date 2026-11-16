"""
src/assistant_llm.py - Synthèse explicative, chat contextuel et recettes.
Calcule d'abord les repères officiels OMS avant toute reformulation.
"""

import os
import json
import math
import requests
import numpy as np

# Import du comparateur d'achat
from src.comparateur import arbitrer_achat_produits, generer_texte_comparaison_humain

# Repères nutritionnels journaliers recommandés pour un adulte (OMS / ANSES)
REPERES_JOURNALIERS_OMS = {
    "sel_g": 5.0,        # Max 5g de sel par jour
    "sucres_g": 25.0,    # Max 25g de sucres libres par jour
    "graisses_g": 70.0   # Environ 70g de lipides par jour
}

BESOIN_CALORIQUE_JOURNALIER = 2000.0  # kcal


def analyser_calories_produit(produit: dict, portion_g: float = None) -> dict:
    champs = produit.get("champs") or {}
    off = produit.get("off") or {}

    cal_100 = None
    for val in [champs.get("energie_kcal"), off.get("energie_kcal")]:
        if val not in [None, "inconnu", "None", "", "nan", np.nan]:
            try:
                c = float(str(val).replace("kcal", "").replace(",", ".").strip())
                if not math.isnan(c):
                    cal_100 = c
                    break
            except (ValueError, TypeError):
                pass

    if cal_100 is None or math.isnan(cal_100):
        return {
            "disponible": False,
            "cal_100g": 0.0,
            "portion_g": 0.0,
            "cal_portion": 0.0,
            "pct_portion": 0,
            "pct_100g": 0,
            "phrase_chatbot": "ℹ️ **Calories non renseignées :** Aucune valeur énergétique lisible sur cette étiquette."
        }

    cat = str(off.get("categories", "")).lower()
    if portion_g is None:
        if any(b in cat for b in ["boisson", "jus", "soda", "eau"]):
            portion_g = 200.0
        elif any(s in cat for s in ["snack", "chips", "biscuit", "chocolat"]):
            portion_g = 30.0
        else:
            portion_g = 100.0

    cal_portion = (cal_100 * portion_g) / 100.0
    pct_portion = int((cal_portion / 2000.0) * 100) if not math.isnan(cal_portion) else 0
    pct_100g = int((cal_100 / 2000.0) * 100) if not math.isnan(cal_100) else 0

    if cal_100 == 0:
        phrase = "💧 **Zéro calorie vérifié :** Ce produit n'apporte aucune énergie."
    elif cal_100 > 400:
        phrase = f"⚡ **Forte densité calorique :** **{cal_100:.0f} kcal / 100g** ({pct_portion}% de vos besoins quotidiens par portion de {portion_g:.0f}g)."
    else:
        phrase = f"⚡ **Apport calorique :** **{cal_100:.0f} kcal / 100g** (soit **{cal_portion:.0f} kcal** par portion de {portion_g:.0f}g)."

    return {
        "disponible": True,
        "cal_100g": cal_100,
        "portion_g": portion_g,
        "cal_portion": cal_portion,
        "pct_portion": pct_portion,
        "pct_100g": pct_100g,
        "phrase_chatbot": phrase
    }


API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or ""


def _appeler_llm(messages: list, temperature: float = 0.3) -> str | None:
    if not API_KEY:
        return None

    try:
        url = "https://api.groq.com/openai/v1/chat/completions" if API_KEY.startswith("gsk_") else "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-8b-instant" if API_KEY.startswith("gsk_") else "gpt-4o-mini",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 350
        }
        rep = requests.post(url, headers=headers, json=payload, timeout=8)
        if rep.status_code == 200:
            return rep.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return None


def generer_bilan_pedagogique_humain(nom: str, energie: float, gras: float, sucre: float, sel: float, prot: float, profil: dict) -> str:
    tous_vides = all(v is None or (isinstance(v, float) and math.isnan(v)) for v in [energie, gras, sucre, sel, prot])
    if tous_vides:
        return (
            f"### 🧑‍⚕️ L'Avis de NutriScan AI pour **{nom}**\n\n"
            "ℹ️ **Informations nutritionnelles non visibles sur cette photo :**\n"
            "Cette face du conditionnement ne contient pas de tableau nutritionnel lisible (calories, sucres, sel).\n\n"
            "👉 **Conseil pratique :** Prenez en photo le dos de la bouteille où figure le tableau des valeurs pour 100ml, ou tapez directement son code-barres dans les options avancées pour obtenir le bilan complet avec les repères de l'OMS !"
        )

    lignes = [f"### 📋 Bilan Nutritionnel Pédagogique : **{nom}**\n"]

    if sucre is not None and not (isinstance(sucre, float) and math.isnan(sucre)):
        pct_sucre = (sucre / 25.0) * 100
        if sucre > 12.5:
            lignes.append(f"• 🍭 **Sucres ({sucre:.1f}g / 100g) :** Teneur élevée. 100g couvrent déjà **{pct_sucre:.0f}%** de la limite journalière recommandée par l'OMS (25g).")
        else:
            lignes.append(f"• 🟢 **Sucres ({sucre:.1f}g / 100g) :** Apport modéré ({pct_sucre:.0f}% du repère OMS).")

    if sel is not None and not (isinstance(sel, float) and math.isnan(sel)):
        pct_sel = (sel / 5.0) * 100
        if sel > 1.5:
            lignes.append(f"• 🧂 **Sel ({sel:.2f}g / 100g) :** Teneur élevée, équivalent à **{pct_sel:.0f}%** du maximum OMS recommandé par jour (5g).")
        else:
            lignes.append(f"• 🟢 **Sel ({sel:.2f}g / 100g) :** Teneur faible et bien maîtrisée.")

    if gras is not None and not (isinstance(gras, float) and math.isnan(gras)):
        pct_gras = (gras / 70.0) * 100
        if gras > 20.0:
            lignes.append(f"• 🧈 **Graisses ({gras:.1f}g / 100g) :** Aliment lipidique riche ({pct_gras:.0f}% des apports de référence).")
        else:
            lignes.append(f"• 🟢 **Graisses ({gras:.1f}g / 100g) :** Teneur en lipides équilibrée.")

    if prot is not None and not (isinstance(prot, float) and math.isnan(prot)) and prot >= 8.0:
        lignes.append(f"• 🥩 **Protéines ({prot:.1f}g / 100g) :** Bonne teneur en protéines.")

    lignes.append("\n**🎯 Adéquation avec votre santé :**")
    alertes_profil = []
    if profil.get("sans_porc"):
        alertes_profil.append("Garantie sans porc / Halal active.")
    if profil.get("peu_de_sel") and sel and not math.isnan(sel) and sel > 1.0:
        alertes_profil.append("Vigilance sel pour votre tension artérielle.")
    if profil.get("peu_de_sucre") and sucre and not math.isnan(sucre) and sucre > 10.0:
        alertes_profil.append("Vigilance sur les sucres rapides (Diabète/Glycémie).")
    if profil.get("peu_de_gras") and gras and not math.isnan(gras) and gras > 15.0:
        alertes_profil.append("Vigilance lipides (Gestion du cholestérol).")

    if alertes_profil:
        for a in alertes_profil:
            lignes.append(f"• ⚠️ {a}")
    else:
        lignes.append("• ✅ Aucun facteur d'incompatibilité critique identifié avec votre profil actuel.")

    return "\n".join(lignes)


def repondre_chat_avec_contexte(question: str, produit_courant: dict, produit_precedent: dict, profil: dict) -> str:
    """
    Répond aux questions sur le produit et déclenche le moteur de comparaison d'achat.
    """
    if not produit_courant:
        return "ℹ️ Aucun produit scanné en mémoire. Téléversez une photo pour commencer !"

    nom_courant = (produit_courant.get("off") or {}).get("nom") or produit_courant.get("nom", "Produit scanné")
    nutr_courant = produit_courant.get("champs") or produit_courant.get("off") or {}
    q_lower = question.lower()

    # 1. Question sur les calories / énergie
    if any(c in q_lower for c in ["calorie", "calorique", "kcal", "energie", "grossir"]):
        bilan = analyser_calories_produit(produit_courant)
        if not bilan["disponible"]:
            return f"📊 **Bilan calorique pour {nom_courant} :**\n\n{bilan['phrase_chatbot']}"
        return (
            f"📊 **Bilan calorique pour {nom_courant} :**\n\n"
            f"• Pour 100g : **{bilan['cal_100g']:.0f} kcal** ({bilan['pct_100g']}% de vos besoins journaliers).\n"
            f"• Par portion de {bilan['portion_g']:.0f}g : **{bilan['cal_portion']:.0f} kcal** ({bilan['pct_portion']}%).\n\n"
            f"{bilan['phrase_chatbot']}"
        )

    # 2. Question sur les ingrédients
    if "ingred" in q_lower or "composition" in q_lower or "contient" in q_lower:
        ingr_txt = (
            (produit_courant.get("document_universel") or {}).get("ingredients_text")
            or (produit_courant.get("off") or {}).get("ingredients_text")
            or (produit_courant.get("champs") or {}).get("ingredients")
            or "Liste d'ingrédients non disponible sur cette étiquette."
        )
        return f"📜 **Ingrédients identifiés pour {nom_courant} :**\n\n« {ingr_txt} »"

    # 3. Question sur la santé / l'adéquation
    if any(k in q_lower for k in ["healt", "adequat", "adéquat", "bon pour", "avis", "sante", "santé", "danger"]):
        score_comp = (produit_courant.get("eval_compatibilite") or {}).get("score", 50)
        positives = (produit_courant.get("eval_compatibilite") or {}).get("positives", [])
        negatives = (produit_courant.get("eval_compatibilite") or {}).get("negatives", [])
        alerte = (produit_courant.get("eval_compatibilite") or {}).get("alerte")

        if alerte:
            return f"🚨 **Verdict : Déconseillé pour vous !**\n\n⚠️ {alerte}"
        elif score_comp >= 70:
            recap = "\n• " + "\n• ".join(positives) if positives else "Valeurs conformes à vos objectifs."
            return f"✅ **Verdict : Ce produit est adéquat pour votre profil (Score : {score_comp}/100)**.\n\nPoints forts :{recap}"
        else:
            recap = "\n• " + "\n• ".join(negatives) if negatives else "Contient des nutriments à limiter."
            return f"⚠️ **Verdict : Produit à consommer avec vigilance (Score : {score_comp}/100)**.\n\nPoints de vigilance :{recap}"

    # 4. DUEL / COMPARAISON ENTRE LES DEUX PRODUITS
    if any(m in q_lower for m in ["compare", "comparer", "comparaison", "précédent", "precedent", "lequel choisir", "lequel acheter", "lequel est mieux"]):
        if not produit_precedent:
            return "ℹ️ Je n'ai qu'un seul aliment en mémoire. **Scannez un deuxième produit** pour lancer le duel comparatif !"
        bilan = arbitrer_achat_produits(produit_courant, produit_precedent, profil)
        return generer_texte_comparaison_humain(bilan)

    # 5. Question ouverte LLM
    if API_KEY:
        nom_prec = (produit_precedent.get("off") or {}).get("nom", "Aucun") if produit_precedent else "Aucun"
        contexte = f"""
        Produit analysé : {nom_courant} (Nutriments: {nutr_courant}).
        Produit précédent : {nom_prec}.
        Profil utilisateur : {profil}.
        Question : {question}
        """
        messages = [
            {"role": "system", "content": "Tu es un assistant nutritionniste concis et bienveillant. Réponds en 2 à 3 phrases claires avec les nutriments du produit."},
            {"role": "user", "content": contexte}
        ]
        rep = _appeler_llm(messages)
        if rep:
            return rep

    return f"Je dispose des informations pour **{nom_courant}**. Vous pouvez m'interroger sur ses calories, ses ingrédients ou son adéquation avec votre profil !"


def suggerer_recettes_saines(produit: dict, profil: dict) -> list[dict]:
    nom = (produit.get("off") or {}).get("nom") or produit.get("nom", "Aliment")
    cat = str((produit.get("off") or {}).get("categories", "")).lower()

    if API_KEY:
        messages = [
            {"role": "system", "content": (
                "Tu es un chef cuisinier passionné de nutrition saine. Propose 2 idées de repas simples au format JSON : "
                '[{"titre": "...", "temps": "15 min", "description": "...", "atout_sante": "..."}]'
            )},
            {"role": "user", "content": f"Produit scanné : {nom} (Catégorie : {cat}). Profil : {profil}"}
        ]
        try:
            rep = _appeler_llm(messages, temperature=0.5)
            if rep:
                debut = rep.find("[")
                fin = rep.rfind("]") + 1
                return json.loads(rep[debut:fin])
        except Exception:
            pass

    if any(m in cat or m in nom.lower() for m in ["lait", "yaourt", "raibi", "fromage"]):
        return [
            {
                "titre": "Bowl énergie aux fruits frais et graines de chia",
                "temps": "5 min",
                "description": f"Mélangez votre {nom} avec une poignée de fruits rouges et des flocons d'avoine.",
                "atout_sante": "Apporte des fibres pour réguler la glycémie."
            },
            {
                "titre": "Smoothie protéiné sans sucres ajoutés",
                "temps": "5 min",
                "description": f"Mixez {nom} avec une banane et une pointe de cannelle.",
                "atout_sante": "Riche en calcium et sans sucre raffiné."
            }
        ]
    elif any(m in cat or m in nom.lower() for m in ["knorr", "epice", "assaisonnement", "sauce"]):
        return [
            {
                "titre": "Poêlée méditerranéenne de légumes croquants",
                "temps": "15 min",
                "description": f"Courgettes et pois chiches sautés avec une touche de {nom}.",
                "atout_sante": "Diminue la concentration en sel par portion."
            },
            {
                "titre": "Papillote de poisson blanc aux herbes",
                "temps": "20 min",
                "description": f"Filet de cabillaud assaisonné avec citron et un soupçon de {nom}.",
                "atout_sante": "Riche en oméga-3."
            }
        ]
    else:
        return [
            {
                "titre": f"Assiette équilibrée autour de {nom}",
                "temps": "10 min",
                "description": "Accompagnez une petite portion avec une belle salade fraîche de crudités.",
                "atout_sante": "Les fibres des crudités ralentissent l'absorption des lipides et glucides."
            }
        ]