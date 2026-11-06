# Compte-Rendu d'Étape : Données, ETL et EDA
**Projet :** Lecture d’étiquettes alimentaires et moteur de recherche explicatif  

## 1. Objectif de l'étape S3
L'objectif  est de mettre en place le socle de données nécessaire au projet :
1. **Collecter et nettoyer (ETL)** un corpus représentatif de 100 à 250 produits alimentaires depuis l'API Open Food Facts.
2. **Réaliser l'analyse exploratoire (EDA)** pour vérifier la distribution des nutriments et la disponibilité des informations sans inventer de métriques sanitaires.
3. **Constituer la partition de test** (les 40 étiquettes d'évaluation requises par le cahier des charges).
4. **Initialiser l'index documentaire** pour préparer le moteur de recherche explicatif (RAG) .



## 2. Collecte des Données (Extraction)

Pour constituer notre jeu de données, nous avons développé le script `src/scripts/clean_dataset.py` qui interroge l'API web d'Open Food Facts.

### Caractéristiques de la collecte :
* **Requêtes paginées :** Téléchargement de 5 pages de 50 produits, triés par popularité (`unique_scans_n`), avec un User-Agent personnalisé identifiant notre projet étudiant.
* **Volume brut récupéré :** Environ 200 produits combinant des références locales marocaines (*Sidi Ali, Jben Jaouda, Cosumar, Danino*) et internationales (*Nutella, Barilla, Coca-Cola, Lindt*).
* **Champs collectés :**
  * Identifiants : `code` (code-barres), `product_name`.
  * Catégorisation : `categories`, `nutriscore_grade`.
  * Données textuelles & visuelles : `ingredients_text`, `image_url`.
  * Valeurs nutritionnelles pour 100g : `energy_kcal`, `fat_100g`, `sugars_100g`, `proteins_100g`, `salt_100g`.

---

## 3. Nettoyage et Transformation (Pipeline ETL)

Le traitement des données brutes a suivi 5 règles strictes de qualité des données :

### 3.1 Dédoublonnage et typage
* Suppression des doublons basés sur la clé primaire unique `code`.
  On a identifié les doublons grâce à la clé primaire unique du produit : son code-barres (EAN). Dans le script ETL, nous avons appliqué df.drop_duplicates(subset=['code']) après avoir nettoyé la chaîne de caractères.
  Dans Open Food Facts, une même référence peut être scannée plusieurs fois ou apparaître sur plusieurs pages de recherche. Supprimer les doublons évite de biaiser nos analyses statistiques (EDA) et d'alourdir inutilement notre index documentaire pour le RAG.

* Conversion systématique des valeurs nutritionnelles en nombres décimaux (`float`).

### 3.2 Gestion des valeurs aberrantes (Sanity Checks)
* Les nutriments pour 100g (`fat`, `sugars`, `proteins`, `salt`) doivent être compris entre $0\text{g}$ et $100\text{g}$.
* L'énergie par 100g est plafonnée à un maximum physique de $950\text{ kcal}$.
* **Choix méthodologique :** Lorsqu'une valeur est aberrante, elle est remplacée par `NaN` au lieu de supprimer toute la ligne, afin de préserver les autres informations valides du produit.
Open Food Facts étant une base collaborative, les internautes font parfois des erreurs de frappe (ex. taper 500g au lieu de 0.5g de sel). En remplaçant uniquement la valeur erronée par NaN, on assainit les statistiques sans perdre le produit, son image, ni sa liste d'ingrédients. 

### 3.3 Traitement des valeurs manquantes (Sans imputation arbitraire)
* **Nutriments numériques :** Nous avons choisi de **laisser les valeurs manquantes à `NaN`**. L'imputation par la moyenne ou la médiane a été expressément évitée, car elle attribuerait par exemple 15g de sucre à une eau minérale. De plus, préserver ces manques permet de tester la capacité de l'OCR à combler les lacunes de la base de données.

* **Champs textuels :** Les noms manquants ont été complétés par `"Produit sans nom"`, les catégories par `"Non classé"` et les Nutri-Scores absents par `"inconnu"` (norme officielle Open Food Facts).
* **Ingrédients :** Suppression des balises Markdown (`_blé_` $\rightarrow$ `blé`) et des retours à la ligne intempestifs pour faciliter l'analyse NLP.

### 3.4 Traçabilité
Une colonne `source` valant `"Open Food Facts API"` a été ajoutée pour documenter l'origine de chaque enregistrement.

* **Résultat du pipeline :** **199 produits propres** enregistrés dans `data/openfoodfacts_clean.csv`.

---

## 4. Analyse Exploratoire des Données (EDA)

L'exécution du script `src/scripts/eda.py` a permis de générer trois figures clés d'analyse :

### 4.1 Complétude des champs (`eda_valeurs_manquantes.png`)
* Les champs d'identification et de texte (`code`, `categories`, `ingredients_text`) affichent **0 % de valeurs manquantes**.
* Les nutriments (`sugars_100g`, `salt_100g`, `energy_kcal`) présentent un taux d'absence d'environ **5,5 % à 6 %**.
* *Interprétation :* Ces manques correspondent principalement aux eaux minérales ou au café soluble, qui ne disposent légalement d'aucune déclaration nutritionnelle obligatoire.

### 4.2 Équilibre du Nutri-Score (`eda_nutriscore.png`)
* La répartition couvre de façon équilibrée l'ensemble des classes officielles :
  * Grade E : ~45 produits
  * Grade C : ~43 produits
  * Grade D : ~35 produits
  * Grade B : ~34 produits
  * Grade A : ~34 produits
  * Inconnu : 8 produits
* *Interprétation :* Le dataset ne présente pas de biais vers une catégorie unique d'aliments et représente fidèlement la diversité des produits du marché.

### 4.3 Distributions nutritionnelles (`eda_distributions.png`)
* **Sucres :** Distribution asymétrique droite typique. Une forte concentration de produits contient moins de 10g de sucre, complétée par une traîne allant jusqu'à 60–100g (confiseries, confitures, pâtes à tartiner).
* **Sel :** La majorité écrasante des produits se situe sous les 2g pour 100g. Quelques valeurs extrêmes isolées ont été identifiées (sel de table pur ou contributions erronées d'utilisateurs), justifiant pleinement l'utilité d'un comparateur automatique OCR/Base pour lever les incohérences.

---

## 5. Partitions & Index Documentaire

Pour clore les exigences de la semaine S3 :

1. **Partition des 40 étiquettes (Ground Truth) :**
   * Via le script `creer_benchmark_40.py`, nous avons isolé 40 produits variés disposant à la fois d'une liste d'ingrédients complète et d'une image d'emballage nette.
   * Les 40 photographies sont téléchargées dans `data/images_benchmark_40/` et associées au fichier `data/ground_truth_40.csv` pour servir d'étalon de mesure lors de l'évaluation OCR.

2. **Index Documentaire (RAG) :**
   * Constitution d'un corpus documentaire officiel composé de recommandations nutritionnelles et de fiches additifs (sources : **ANSES**, **EFSA**, **DGCCRF**, **Santé Publique France**).
   * Les textes ont été vectorisés avec le modèle multilingue léger `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
   * Un seuil de similarité cosinus ($0.35$) a été implémenté pour forcer le statut *« Sources insuffisantes »* en cas de question hors-domaine afin d'éviter les hallucinations.

---

## 6. Conclusion et État d'avancement
À l'issue de cette étape S3 :
* Le corpus de **199 produits** est assaini, tracé et documenté.
* Les visualisations statistiques confirment la qualité et la représentativité de l'échantillon.
* Les **40 étiquettes étalon** et **l'index documentaire** sont prêts pour aborder l'étape **S4 (Baseline d'extraction OCR et premier retrieval RAG)**.