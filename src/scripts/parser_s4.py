import re
import unicodedata
import numpy as np


def normaliser(texte: str) -> str:
    """
    Nettoie le texte :
    - Supprime les accents français sans effacer l'arabe (diacritiques uniquement)
    - Minuscules
    - Harmonise les apostrophes et tirets typiques de l'OCR
    """
    if not texte:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(texte))
    texte_sans_accent = "".join(c for c in nfkd if not unicodedata.combining(c))
    texte = texte_sans_accent.lower()
    texte = texte.replace("’", "'").replace("–", "-").replace("—", "-")
    return texte


def convertir_en_flottant(val_str: str, unite_str: str = "") -> float:
    """Convertit une chaîne en float et gère les mg -> g."""
    if not val_str:
        return np.nan
    try:
        val = float(val_str.replace(",", ".").strip())
        if unite_str and "mg" in unite_str.lower():
            val = val / 1000.0
        return val
    except (ValueError, TypeError):
        return np.nan


def chercher_nombre(motif_cle: str, texte: str) -> float:
    """
    Cherche un nombre associé à une clé nutritionnelle, 
    soit sur la même ligne, soit immédiatement sur la ligne suivante.
    """
    regex = rf"\b(?:{motif_cle})\b[^\d\n]*(?:\n[^\d\n]*)?(\d+[\.,]?\d*)\s*(mg|g)?\b"
    m = re.search(regex, texte)
    if m:
        return convertir_en_flottant(m.group(1), m.group(2) or "")
    return np.nan


def extraire_energie(texte: str) -> float:
    """Extrait l'énergie en kcal (physiquement plafonnée à 950 kcal/100g)."""
    # 1. Chiffre immédiatement collé ou précédé de kcal
    m_kcal = re.search(r"(\d+[\.,]?\d*)\s*kcal\b", texte)
    if m_kcal:
        val = convertir_en_flottant(m_kcal.group(1))
        if 0 <= val <= 950:
            return val

    # 2. Cas classique étiquette : 'energie ... kj / ... kcal'
    m_slash = re.search(r"\benergie\b[^\n]*?/\s*(\d+[\.,]?\d*)\s*kcal?", texte)
    if m_slash:
        val = convertir_en_flottant(m_slash.group(1))
        if 0 <= val <= 950:
            return val

    # 3. Mot clé énergie avec saut de ligne possible (filtre strict < 950 pour éviter les kJ)
    val = chercher_nombre(r"energie|energy|calories?", texte)
    if not np.isnan(val) and 0 <= val <= 950:
        return val

    return np.nan


def extraire_graisses(texte: str) -> float:
    """Extrait les matières grasses totales (0-100g)."""
    val = chercher_nombre(r"matieres\s*grasses|total\s*fat|\bfat\b|lipides?", texte)
    return val if (not np.isnan(val) and 0 <= val <= 100) else np.nan


def extraire_sucres(texte: str) -> float:
    """Extrait les sucres (priorité à 'dont sucres' pour ne pas confondre avec glucides)."""
    # Priorité 1 : "dont sucres" / "of which sugars"
    val_dont = chercher_nombre(r"dont\s*sucres|of\s*which\s*sugars", texte)
    if not np.isnan(val_dont) and 0 <= val_dont <= 100:
        return val_dont

    # Priorité 2 : mot sucre(s) seul
    val_sucre = chercher_nombre(r"sucres?|sugars?", texte)
    if not np.isnan(val_sucre) and 0 <= val_sucre <= 100:
        return val_sucre

    return np.nan


def extraire_proteines(texte: str) -> float:
    """Extrait les protéines (0-100g)."""
    val = chercher_nombre(r"proteines?|proteins?|eiweiss", texte)
    return val if (not np.isnan(val) and 0 <= val <= 100) else np.nan


def extraire_sel(texte: str) -> float:
    """Extrait le sel (avec conversion si seul le sodium est mentionné)."""
    val_sel = chercher_nombre(r"sel|salt|salz", texte)
    if not np.isnan(val_sel) and 0 <= val_sel <= 100:
        return val_sel

    # Si sodium seul : sel = sodium * 2.54
    val_sod = chercher_nombre(r"sodium", texte)
    if not np.isnan(val_sod):
        sel_estime = round(val_sod * 2.54, 2)
        if 0 <= sel_estime <= 100:
            return sel_estime

    return np.nan


def extraire_ingredients(texte: str) -> str:
    """
    Extrait les ingrédients avec tolérance maximale aux fautes OCR :
    - Tolère 'ingredients', 'igredients' (sans N), 'redients' (début coupé), 'ingredie'
    - Supprime les deux-points et résidus d'en-tête
    - Arrêt propre dès les mentions légales
    """
    patterns_fin = (
        r"(traces|allergenes|valeurs|nutrition|conservation|"
        r"poids\s*net|service|fabrique|a\s*consommer|mode\s*d|conseil|$)"
    )

    # Motif robuste qui attrape toutes les variantes et coquilles OCR courantes
    pattern_debut = r"(?:(?:i[n]?)?g?r[eé]die(?:nts?|s)?|ingred\w*)"
    regex = rf"\b{pattern_debut}\s*[:\.]?\s*(.+?)(?={patterns_fin})"

    m = re.search(regex, texte, re.DOTALL)
    if m:
        nettoye = m.group(1).strip()
        # Suppression des puces, résidus Markdown et caractères parasites
        nettoye = re.sub(r"[\*\_•\t\:\.]", " ", nettoye)
        nettoye = re.sub(r"\s+", " ", nettoye).strip()
        if len(nettoye) >= 3:
            return nettoye

    return "inconnu"


def parser_etiquette(texte_brut_ocr: str) -> dict:
    """Fonction principale recevant le texte OCR brut et renvoyant le dictionnaire propre."""
    texte = normaliser(texte_brut_ocr)
    return {
        "energie_kcal": extraire_energie(texte),
        "graisses": extraire_graisses(texte),
        "sucres": extraire_sucres(texte),
        "proteines": extraire_proteines(texte),
        "sel": extraire_sel(texte),
        "ingredients": extraire_ingredients(texte)
    }