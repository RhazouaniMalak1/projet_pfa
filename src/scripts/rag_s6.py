import os
from pathlib import Path
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# 1. Chargement du modèle
print("⏳ Chargement du modèle d'embeddings sémantiques...")
modele = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 2. Corpus officiel certifié
CORPUS = [
    {
        "titre": "Recommandations sur le Sel",
        "url": "https://www.anses.fr/fr/content/le-sel",
        "texte": "L'OMS recommande de consommer moins de 5g de sel par jour (environ 2g de sodium) pour les adultes. Un excès de sel favorise l'hypertension artérielle et augmente les risques de maladies cardiovasculaires. Les principales sources de sel sont le pain, les charcuteries, les fromages et les plats préparés industriels."
    },
    {
        "titre": "Recommandations sur les Sucres",
        "url": "https://www.mangerbouger.fr/les-recommandations/reduire-le-sucre",
        "texte": "L'ANSES recommande de ne pas consommer plus de 100g de sucres totaux par jour (hors lactose et galactose), et de limiter au maximum les boissons sucrées. Une consommation excessive de sucres libres favorise la prise de poids, l'obésité, le diabète de type 2 et les caries dentaires."
    },
    {
        "titre": "Additifs : Nitrites et E250",
        "url": "https://www.anses.fr/fr/content/nitrites-et-nitrates",
        "texte": "Le nitrite de sodium (E250) est un conservateur très utilisé dans les charcuteries pour empêcher le développement de la bactérie Clostridium botulinum. Cependant, l'ANSES confirme un lien entre l'exposition aux nitrites/nitrates et un risque accru de cancer colorectal. Il est recommandé de limiter sa consommation de charcuterie à 150g par semaine."
    },
    {
        "titre": "Comprendre le Nutri-Score",
        "url": "https://www.santepubliquefrance.fr/nutriscore",
        "texte": "Le Nutri-Score est un logo à 5 couleurs allant de A (vert, meilleure qualité nutritionnelle) à E (orange foncé, moins favorable). Il prend en compte les éléments à favoriser (fibres, protéines, fruits, légumes, légumineuses) et les éléments à limiter (calories, acides gras saturés, sucres simples, sel pour 100g)."
    },
    {
        "titre": "Les 14 Allergènes Majeurs",
        "url": "https://www.economie.gouv.fr/dgccrf/allergie-alimentaire",
        "texte": "La réglementation européenne impose l'étiquetage obligatoire de 14 allergènes majeurs : céréales contenant du gluten, crustacés, œufs, poissons, arachides, soja, lait (lactose), fruits à coque, céleri, moutarde, graines de sésame, sulfites, lupin et mollusques. Ils doivent apparaître en gras sur l'emballage."
    },
    {
        "titre": "Additif E322 : Lécithines",
        "url": "https://www.anses.fr",
        "texte": "L'additif E322 correspond aux lécithines, le plus souvent extraites du soja ou du tournesol. C'est un émulsifiant naturel qui permet de lier l'eau et les graisses dans les chocolats et biscuits. Il est considéré comme sans danger toxicologique par l'EFSA, mais les personnes allergiques au soja doivent rester vigilantes."
    },
    {
        "titre": "Additif E330 : Acide citrique",
        "url": "https://www.efsa.europa.eu",
        "texte": "Le E330 (acide citrique) est un acidifiant et antioxydant naturel présent dans les agrumes, souvent produit industriellement par fermentation. Il est totalement inoffensif pour l'organisme selon l'EFSA, bien qu'une consommation très excessive puisse favoriser l'érosion de l'émail dentaire."
    },
    {
        "titre": "Additif E500 et E503 : Poudres à lever",
        "url": "https://www.anses.fr",
        "texte": "Les additifs E500 (carbonates de sodium) et E503 (carbonates d'ammonium) sont des poudres à lever indispensables en biscuiterie pour faire gonfler la pâte à la cuisson. Ils se décomposent à la chaleur et sont reconnus sans aucun risque pour la santé humaine."
    },
    {
        "titre": "Additif E150d : Caramel au sulfite d'ammoniaque",
        "url": "https://www.anses.fr",
        "texte": "Le E150d est un colorant caramel de synthèse utilisé dans les sodas au cola et sauces. Bien qu'autorisé, il contient des sous-produits de fabrication (4-MEI). L'ANSES et l'EFSA ont fixé une dose journalière admissible stricte car une exposition très élevée est suspectée d'effets cancérigènes."
    }
]

# 3. Vrai Chunking avec découpage par phrases
def decouper_texte(texte: str, source: str, max_mots: int = 60, chevauchement: int = 15):
    mots = texte.split()
    if len(mots) <= max_mots:
        return [{"texte": texte.strip(), "source": source}]
    
    passages = []
    i = 0
    while i < len(mots):
        chunk = " ".join(mots[i:i + max_mots])
        passages.append({"texte": chunk, "source": source})
        i += (max_mots - chevauchement)
    return passages

tous_les_passages = []
for doc in CORPUS:
    source_nom = f"{doc['titre']} ({doc['url']})"
    tous_les_passages.extend(decouper_texte(doc["texte"], source_nom))

# 4. Indexation et mise en cache disque
chemin_cache = Path("data/rag_embeddings.npy")
chemin_cache.parent.mkdir(parents=True, exist_ok=True)

if chemin_cache.exists():
    vecteurs = np.load(chemin_cache)
else:
    print(f"📦 Calcul et sauvegarde des embeddings pour {len(tous_les_passages)} passages...")
    vecteurs = modele.encode([p["texte"] for p in tous_les_passages], normalize_embeddings=True)
    np.save(chemin_cache, vecteurs)

# 5. Recherche Sémantique avec filtrage par seuil
def chercher(question: str, passages=tous_les_passages, vecteurs_corpus=vecteurs, k=2, seuil=0.35):
    q_vec = modele.encode([question], normalize_embeddings=True)[0]
    scores = vecteurs_corpus @ q_vec
    meilleurs_indices = np.argsort(scores)[::-1][:k]

    resultats = []
    for idx in meilleurs_indices:
        score = float(scores[idx])
        if score >= seuil:
            resultats.append((passages[idx], score))
    return resultats

# 6. Recherche Hybride spécifique aux additifs (Anti-Hallucination stricte)
def expliquer_additif_rag(code_e: str) -> dict:
    code_propre = code_e.strip().upper()
    
    # 1. Vérification par mot-clé exact d'abord (Filtre déterministe)
    passages_pertinents = []
    for p in tous_les_passages:
        if re.search(rf"\b{code_propre}\b", p["texte"], re.IGNORECASE) or re.search(rf"\b{code_propre}\b", p["source"], re.IGNORECASE):
            passages_pertinents.append(p)
            
    if passages_pertinents:
        return {
            "explication": passages_pertinents[0]["texte"],
            "source": passages_pertinents[0]["source"],
            "confiance": "Exacte (Code vérifié)"
        }

    # 2. Recherche sémantique en repli (Fallback)
    question = f"Quel est le rôle et le niveau de risque de l'additif {code_propre} ?"
    resultats = chercher(question, k=1, seuil=0.45) # Seuil plus strict pour les additifs

    if resultats:
        p, score = resultats[0]
        # Vérifier que le code est bien mentionné dans le texte pour ne pas confondre E100 et E250
        if code_propre.lower() in p["texte"].lower() or code_propre.lower() in p["source"].lower():
            return {
                "explication": p["texte"],
                "source": p["source"],
                "confiance": f"Sémantique (Score: {score:.2f})"
            }

    return {
        "explication": f"L'additif {code_propre} n'est pas documenté dans notre corpus officiel (ANSES/EFSA).",
        "source": "Non répertorié",
        "confiance": "Nulle"
    }

# 7. Formateur de réponse RAG
def repondre_a_la_question(question: str) -> dict:
    resultats = chercher(question, k=2, seuil=0.35)

    print("\n" + "=" * 65)
    print(f"❓ QUESTION : {question}")
    print("=" * 65)

    if not resultats:
        print("❌ RÉPONSE : Sources insuffisantes.")
        return {"reponse": "Sources insuffisantes.", "sources": []}

    print("🔍 PASSAGES TROUVÉS DANS LE CORPUS :")
    passages_texte = []
    sources = []
    for i, (p, score) in enumerate(resultats, start=1):
        print(f"  [{i}] (Score: {score:.2f}) {p['source']}")
        print(f"      « {p['texte']} »\n")
        passages_texte.append(f"[{p['source']}] : {p['texte']}")
        sources.append(p["source"])

    prompt_llm = (
        "Tu es un expert en sécurité alimentaire et nutrition.\n"
        "Réponds à la question uniquement à partir des passages ci-dessous.\n"
        "Cite la source entre crochets après chaque information.\n"
        "Si les passages ne permettent pas de répondre, écris exactement : 'Sources insuffisantes'.\n\n"
        f"Passages :\n" + "\n".join(passages_texte) + f"\n\nQuestion : {question}"
    )

    return {
        "prompt_llm": prompt_llm,
        "passages": passages_texte,
        "sources": sources
    }

# --- TESTS DE VALIDATION ---
if __name__ == "__main__":
    # Test 1 : Recommandation Sel
    repondre_a_la_question("Quelle est la quantité de sel maximale recommandée par jour ?")

    # Test 2 : Question sur un additif présent (E250)
    print("\n🔬 Test Additif Présent (E250) :")
    print(expliquer_additif_rag("E250"))

    # Test 3 : Question sur un additif absent (E100) -> Ne doit SURTOUT PAS confondre avec un autre
    print("\n🔬 Test Additif Absent (E100) :")
    print(expliquer_additif_rag("E100"))

    # Test 4 : Question hors domaine -> Doit refuser
    repondre_a_la_question("Quelle est la capitale de l'Espagne ?")