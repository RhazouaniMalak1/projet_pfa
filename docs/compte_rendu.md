# Compte-Rendu d'Étape : Pipeline de Données, OCR, Machine Learning et RAG

**Projet :** Lecture d’étiquettes alimentaires, Classification, Estimation de prix et Moteur explicatif RAG  
**Auteurs :** Équipe Projet PFA  
**Livrables validés :** 
* `openfoodfacts_clean.csv`, `ground_truth_40.csv`, figures EDA
* `parser_s4.py`, `test_ocr.py` (Moteur PaddleOCR)
* `modele_famille.pkl`, `modele_ml_s5.py` (Classification multiclasse)
* `modele_prix_regresseur.pkl`, `prix_api.py` (Régression & API Hybride de prix)
* `rag_s6.py`, `rag_embeddings.npy`, `assistant_llm.py` (Recherche sémantique & RAG)

---

## 1. Objectifs Globaux du Projet

Le projet a pour ambition de concevoir un assistant nutritionnel intelligent de bout en bout :
1. **Socle de Données (S3) :** Collecter, assainir (ETL) et analyser (EDA) un corpus de produits alimentaires réels via Open Food Facts.
2. **Benchmark & OCR (S4) :** Constituer un étalon de vérité terrain (40 étiquettes) et déployer une chaîne OCR capable d'extraire les nutriments et ingrédients sans hallucination.
3. **Machine Learning Prédictif (S5) :** Classifier les produits par famille alimentaire et estimer une fourchette de prix de marché en Dirhams (MAD).
4. **Moteur RAG & Chatbot Explicatif (S6) :** Fournir des explications scientifiques sourcées (ANSES, EFSA, OMS) et personnalisées au profil de santé de l'utilisateur.

---

## 2. Collecte des Données et Pipeline ETL (Étape S3)

### 2.1 Extraction et multimodalité (`clean_dataset.py`)
Le pipeline interroge l'API web d'Open Food Facts via des requêtes paginées avec un `User-Agent` universitaire dédié.
* **Volume collecté :** Environ 200 références combinant des produits locaux marocains (*Sidi Ali, Jben Jaouda, Raïbi Jamila, Cosumar, Danino*) et internationaux (*Nutella, Barilla, Coca-Cola, Harrys, Lindt*).
* **Données visuelles multimodales :**
  * `image_url` / `image_front_url` : vue d'ensemble du conditionnement.
  * `image_ingredients_url` : gros plan haute résolution sur la liste textuelle des ingrédients.
  * `image_nutrition_url` : gros plan haute résolution sur le tableau nutritionnel (clé de voûte de l'OCR chiffré).

### 2.2 Règles strictes de nettoyage et de qualité des données
1. **Dédoublonnage :** Déduplication stricte sur le code-barres unique EAN (`df.drop_duplicates(subset=['code'])`).
2. **Sanity Checks physiques :** 
   * Nutriments pour 100g bornés sur $[0\text{g}, 100\text{g}]$.
   * Énergie bornée sur $[0\text{ kcal}, 950\text{ kcal}]$.
   * Les valeurs aberrantes sont converties en `NaN` (`np.nan`) au lieu de supprimer la ligne, préservant ainsi le texte et les images.
3. **Absence d'imputation arbitraire :** Préservation des nutriments manquants sous forme de `NaN` pour éviter d'attribuer des sucres à de l'eau minérale et tester la capacité de l'OCR à combler les manques.
4. **Normalisation NLP :** Élimination des balises Markdown (`_blé_` $\rightarrow$ `blé`) et nettoyage des retours à la ligne intempestifs.
* **Résultat ETL :** **199 produits propres et documentés** exportés dans `data/openfoodfacts_clean.csv`.

---

## 3. Analyse Exploratoire des Données (EDA)

L'exécution du script `src/scripts/eda.py` a produit trois synthèses visuelles :

1. **Complétude des données (`eda_valeurs_manquantes.png`) :**
   * Code, nom et ingrédients affichent **0 % d'absence**.
   * Les nutriments présentent un faible taux d'absence d'environ **5,5 % à 6 %**, correspondant légitimement aux denrées brutes exemptées de déclaration obligatoire (café pur, eaux minérales).
2. **Équilibre du Nutri-Score (`eda_nutriscore.png`) :**
   * Répartition harmonieuse sur les classes officielles : Grade E (~45), Grade C (~43), Grade D (~35), Grade B (~34), Grade A (~34), Inconnu (8). L'échantillon ne présente aucun biais catégoriel.
3. **Distributions nutritionnelles (`eda_distributions.png`) :**
   * Sucres : distribution asymétrique droite (médiane $< 10\text{g}$, traîne jusqu'à $80\text{g}$ pour les confiseries).
   * Sel : concentration sous les $2\text{g}/100\text{g}$, justifiant un contrôle de cohérence automatique.

---

## 4. Benchmark Étalon et Extraction OCR (Étape S4)

### 4.1 Constitution du Benchmark 40 (`creer_benchmark_40.py`)
Afin de garantir une évaluation sans faille :
* **Garantie de complétude (40/40) :** Algorithme de filtrage itératif téléchargeant exactement 40 images valides (élimination des liens rompus).
* **Ciblage nutritionnel :** Téléchargement prioritaire des gros plans de tableaux nutritionnels (`image_nutrition_url`) pour permettre l'évaluation chiffrée.
* **Fichiers générés :** Répertoire d'images `data/images_benchmark_40/` couplé au fichier étalon de vérité terrain `data/ground_truth_40.csv`.

### 4.2 Moteur OCR et Parseur Robuste (`parser_s4.py` & `test_ocr.py`)
* **Moteur d'inférence :** Déploiement de **PaddleOCR 3.7** (modèles PP-OCRv6) avec désactivation de l'accélération MKLDNN CPU pour assurer une stabilité parfaite sous Windows sans crash mémoire.
* **Tolérance aux artefacts OCR :**
  * Normalisation Unicode préservant les caractères arabes des produits marocains.
  * Expressions régulières capturant les mots tronqués au cadrage (*ex: « rèdients », « IGRÉDIENTS » sans N, « Ingrédie: »*).
  * Gestion des tableaux bilingues et multi-lignes (*ex: Protéines $\rightarrow$ Eiwitten $\rightarrow$ 5.4g*).
  * Conversion automatique des milligrammes en grammes ($\text{mg} \rightarrow \text{g}$).
* **Validation expérimentale :**
  * Extraction à **100 % d'exactitude** sur l'étiquette de noix Lidl (`20005733.jpg`) : Énergie 712 kcal, Lipides 69.1g, Sucres 0.9g, Protéines 15.5g, Sel 0.02g.
  * Extraction complète des ingrédients sur la Ricoré (`3033710073467.jpg`).
  * Comportement anti-hallucination vérifié : renvoie strictement `NaN` lorsqu'une image ne contient pas de tableau nutritionnel.

---

## 5. Modélisation Machine Learning (Étape S5)

### 5.1 Classification Multiclasse de Familles de Produits (`modele_ml_s5.py`)
* **Objectif :** Prédire automatiquement la famille du produit (*Boissons*, *Biscuits & Snacks*, *Produits Laitiers*, *Épicerie & Plats*).
* **Pipeline étanche :** Intégration d'un `SimpleImputer(strategy="median")` couplé à un `RandomForestClassifier`. Cette architecture garantit que l'application ne plantera jamais, même si l'OCR omet un nutriment.
* **Résultats comparatifs (Test sur 27 produits réels) :**

| Famille Produit | Baseline Naïve (Dummy) | Random Forest Pipeline (F1) | Précision | Rappel |
| :--- | :---: | :---: | :---: | :---: |
| **Boissons** | 0.00 | **1.00 (100 %)** | 1.00 | 1.00 |
| **Biscuits & Snacks** | 0.54 | **0.90** | 0.90 | 0.90 |
| **Épicerie & Plats** | 0.00 | **0.88** | 0.88 | 0.88 |
| **Produits Laitiers** | 0.00 | **0.67** | 0.67 | 0.67 |
| **Accuracy Globale** | **37 %** | **89 % (+52 pts)** | - | - |
| **Macro F1-Score** | **0.14** | **0.86** | - | - |

* Le modèle est sauvegardé et sérialisé dans `data/modele_famille.pkl`.

### 5.2 Régression et API Hybride de Prix en Dirhams (`prix_api.py`)
* **Architecture Hybride :**
  1. Interrogation en temps réel de l'API officielle **Open Prices** pour récupérer les relevés réels en magasin.
  2. En cas d'absence, bascule automatique sur un modèle de secours **Random Forest Regressor** (`modele_prix_regresseur.pkl`).
* **Fourchette de Marché [Min, Estimé, Max] :** Au lieu d'un prix fixe arbitraire, le modèle calcule la distribution des 150 arbres de décision pour générer une fourchette réaliste :
  * Performance du régresseur : $\text{MAE} = 2.24\text{ DH}$ (Gain de $0.81\text{ DH}$ face à la baseline).
  * Exemple de prédiction en production : `Prix Estimé : 9.67 DH` $\rightarrow$ `Fourchette : [7.47 DH – 11.95 DH]`.

---

## 6. Moteur RAG et Assistant Conversationnel (Étape S6)

### 6.1 Moteur RAG Hybride Anti-Hallucination (`rag_s6.py`)
* **Indexation sémantique :** Vectorisation d'un corpus certifié (**ANSES**, **EFSA**, **DGCCRF**, **Santé Publique France**) via `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
* **Mise en cache :** Sauvegarde des embeddings dans `data/rag_embeddings.npy` pour un temps de réponse instantané.
* **Validation des Garde-Fous :**
  * *Test Sémantique :* Question sur le sel $\rightarrow$ Passage ANSES extrait avec un score élevé de **0.77**.
  * *Test Additif Connu (E250) :* Recherche exacte par mot-clé $\rightarrow$ Fiche Nitrites extraite avec **Confiance : Exacte**.
  * *Test Additif Inconnu (E100) :* Renvoie *"Non répertorié"* $\rightarrow$ **Zéro confusion avec les autres additifs**.
  * *Test Hors-Domaine (« Capitale de l'Espagne ? ») :* Refus strict avec la mention **« Sources insuffisantes »** (score cosinus $< 0.35$).

### 6.2 Assistant LLM et Synthèse Nutritionnelle (`assistant_llm.py`)
* **Architecture Math-First :** Les dépassements de seuils (5g de sel, 25g de sucre selon l'OMS) sont calculés de manière déterministe en Python avant toute génération de texte.
* **Fonctionnalités validées :**
  * Synthèse calorique par portion adaptée à l'aliment (200ml pour boisson, 30g pour snack).
  * Chat comparatif avec mémoire contextuelle (*« Lequel est meilleur entre ce produit et le précédent ? »*).
  * Suggestion de 2 recettes saines conformes aux contraintes alimentaires (sans porc, sans alcool, respect des allergènes).
  * Fonctionnement garanti à 100 % en local même en l'absence de clé API LLM.

---

## 7. Synthèse Générale et État d'Avancement

| Étape | Composant Technique | Statut | Métrique Clé / Livrable |
| :--- | :--- | :---: | :--- |
| **S3** | ETL & Traçabilité Open Food Facts | **Validé** | 199 produits assainis (`openfoodfacts_clean.csv`) |
| **S3** | Benchmark Ground Truth 40 | **Validé** | 40/40 images conformes (`ground_truth_40.csv`) |
| **S4** | Détection OCR & Parsing Regex | **Validé** | 100 % d'extraction sur étiquettes nutritionnelles complètes |
| **S5** | Classification de Famille de Produit | **Validé** | **Accuracy : 89 %**, **Macro F1 : 0.86** (`modele_famille.pkl`) |
| **S5** | Régression & Fourchette de Prix (MAD) | **Validé** | $\text{MAE} = 2.24\text{ DH}$ (`modele_prix_regresseur.pkl`) |
| **S6** | Moteur de Recherche RAG & Additifs | **Validé** | Seuil $0.35$, Zéro hallucination, Cache `.npy` |
| **S6** | Assistant LLM & Recettes Saines | **Validé** | Repères OMS intégrés + Mode déterministe hors-ligne |

**Conclusion :** L'ensemble des modules d'ingénierie des données, de vision par ordinateur (OCR), d'apprentissage automatique (ML) et d'IA générative (RAG) sont testés, interconnectés et prêts pour le déploiement au sein de l'application finale (`app.py`).