# 🛒 E-commerce Analytics: Customer Segmentation & Sales Prediction

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

**Analyse de données e-commerce avec Machine Learning**

 • [Installation](#-installation) • [Résultats](#-résultats) • 

</div>

---

## 📋 Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Contexte du projet](#-contexte-du-projet)
- [Objectifs](#-objectifs)
- [Dataset](#-dataset)
- [Méthodologie](#-méthodologie)
- [Technologies utilisées](#-technologies-utilisées)
- [Installation](#-installation)
- [Structure du projet](#-structure-du-projet)
- [Résultats](#-résultats)
- [Visualisations](#-visualisations)
- [Conclusions et perspectives](#-conclusions-et-perspectives)

---

## 🎯 Vue d'ensemble

Ce projet académique explore l'application du **Machine Learning** dans le domaine du **e-commerce** pour optimiser les stratégies commerciales et améliorer la compréhension du comportement client. En analysant un dataset réel de la plateforme brésilienne **Olist**, nous avons développé des modèles pour :

- 🎯 **Segmenter les clients** en groupes homogènes
- 📈 **Prédire les ventes** futures
- 💡 **Extraire des insights actionnables** pour le business

### 🏆 Réalisations clés

| Métrique | Valeur | Description |
|----------|--------|-------------|
| **Score Silhouette** | 0.72 | Qualité de la segmentation client |
| **Segments identifiés** | 3 | Clusters de clients distincts |
| **R² (Régression)** | 0.127 | Variance expliquée par le modèle |
| **Données traitées** | 99,441 | Commandes analysées (2016-2018) |

---

## 📖 Contexte du projet

### Le défi du e-commerce moderne

Dans un marché e-commerce en constante évolution, les entreprises font face à plusieurs défis :
- 📊 Volume massif de données transactionnelles
- 🎯 Nécessité de personnaliser l'expérience client
- 💰 Optimisation des stratégies marketing et commerciales
- 🔮 Anticipation des tendances de vente

### Notre approche

Ce projet démontre comment le **Machine Learning** peut transformer ces défis en opportunités, en permettant :
- Une compréhension approfondie des segments clients
- Une prédiction des comportements d'achat
- Une optimisation des opérations logistiques

---

## 🎯 Objectifs

### Objectifs principaux

1. **🔍 Analyse exploratoire des données**
   - Comprendre la structure du dataset Olist
   - Identifier les patterns et tendances
   - Détecter et traiter les anomalies

2. **👥 Segmentation des clients**
   - Identifier les profils clients les plus rentables
   - Créer des segments homogènes pour le marketing ciblé
   - Comprendre les comportements d'achat distincts

3. **📊 Prédiction des ventes**
   - Développer des modèles prédictifs pour anticiper les ventes
   - Identifier les facteurs influençant les performances commerciales
   - Optimiser la gestion des stocks et ressources

### Questions de recherche

- ❓ **Quels segments de clients sont les plus rentables ?**
- ❓ **Quels facteurs influencent la satisfaction client ?**
- ❓ **Comment prédire les tendances de vente futures ?**
- ❓ **Quel est l'impact du délai de livraison sur les ventes ?**

---

## 📊 Dataset

### Brazilian E-Commerce Public Dataset by Olist

Le dataset provient de [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) et contient des informations sur **99,441 commandes** passées entre **2016 et 2018** sur plusieurs marketplaces au Brésil.

### 🗂️ Structure des données

| Fichier | Colonnes | Lignes | Description |
|---------|----------|--------|-------------|
| `olist_orders_dataset.csv` | 8 | 99,441 | Informations sur les commandes |
| `olist_order_items_dataset.csv` | 7 | 112,650 | Produits de chaque commande |
| `olist_order_payments_dataset.csv` | 5 | 103,886 | Détails des paiements |
| `olist_customers_dataset.csv` | 5 | 99,441 | Informations clients |
| `olist_products_dataset.csv` | 9 | 32,951 | Catalogue produits |
| `olist_sellers_dataset.csv` | 4 | 3,095 | Informations vendeurs |
| `olist_order_reviews_dataset.csv` | 7 | 99,224 | Avis clients |
| `olist_geolocation_dataset.csv` | 5 | 1,000,163 | Données géographiques |

### 📈 Variables clés

**Variables de commande :**
- `order_status` : Statut de la commande (delivered, shipped, canceled, etc.)
- `order_purchase_timestamp` : Date d'achat
- `order_delivered_customer_date` : Date de livraison

**Variables de paiement :**
- `payment_type` : Mode de paiement (carte, boleto, etc.)
- `payment_value` : Montant du paiement
- `payment_installments` : Nombre de versements

**Variables produit :**
- `product_category_name` : Catégorie du produit
- `price` : Prix du produit
- `freight_value` : Coût de livraison

---

## 🔬 Méthodologie

Notre approche suit un pipeline complet de Data Science :

### 1️⃣ Préparation des données

```
📥 Acquisition → 🔍 Exploration → 🧹 Nettoyage → 🔄 Transformation
```

**Étapes réalisées :**
- ✅ Chargement de 9 fichiers CSV
- ✅ Analyse des valeurs manquantes et outliers
- ✅ Traitement des données manquantes (imputation intelligente)
- ✅ Nettoyage des données textuelles (géolocalisation)
- ✅ Gestion des données déséquilibrées
- ✅ Feature engineering (création de nouvelles variables)

### 2️⃣ Segmentation des clients (Clustering)

**🎯 Algorithme : Clustering Hiérarchique**

**Variables utilisées :**
- Fréquence d'achat
- Panier moyen
- Délai moyen de livraison
- Score moyen des avis clients

**Processus :**
1. Standardisation des données (Z-score)
2. Calcul de la matrice de distances (distance euclidienne)
3. Construction du dendrogramme (méthode de Ward)
4. Détermination du nombre optimal de clusters
5. Évaluation avec le score de Silhouette

**Résultat : 3 segments clients identifiés**

### 3️⃣ Prédiction des ventes (Régression)

**📈 Algorithme : Régression Linéaire Multiple**

**Variables explicatives :**
- Type de paiement (One-Hot Encoding)
- Délai de livraison
- Fréquence d'achat
- Montant des achats précédents

**Variable cible :**
- Montant total payé (`payment_value`)

**Métriques d'évaluation :**
- R² (coefficient de détermination)
- MSE (Mean Squared Error)
- Distribution des erreurs résiduelles

---

## 🛠️ Technologies utilisées

### Langages et Frameworks

```python
# Core Data Science
Python 3.8+          # Langage principal
Pandas 2.1.0         # Manipulation de données
NumPy 1.25.2         # Calculs numériques
```

### Machine Learning

```python
# Scikit-learn
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
```

### Visualisation

```python
Matplotlib 3.7.2     # Graphiques de base
Seaborn 0.12.2       # Visualisations statistiques
```

### Environnement de développement

```
Jupyter Notebook     # Développement interactif
Anaconda            # Gestion d'environnement
```

---

## 📦 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip ou conda installé
- Jupyter Notebook

### Installation rapide

**Option 1 : Avec pip**

```bash
# 1. Cloner le repository
git clone https://github.com/Boutanfitsalma/ecommerce-ml-analysis.git
cd ecommerce-ml-analysis

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter Notebook
jupyter notebook
```

**Option 2 : Avec Conda**

```bash
# 1. Cloner le repository
git clone https://github.com/Boutanfitsalma/ecommerce-ml-analysis.git
cd ecommerce-ml-analysis

# 2. Créer l'environnement conda
conda create -n ecommerce-ml python=3.8
conda activate ecommerce-ml

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter Notebook
jupyter notebook
```

### Téléchargement des données

Les données sont disponibles sur Kaggle :

```bash
# Option 1 : Téléchargement manuel
# Allez sur https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# Téléchargez et décompressez dans le dossier data/

# Option 2 : Avec Kaggle API
pip install kaggle
kaggle datasets download -d olistbr/brazilian-ecommerce
unzip brazilian-ecommerce.zip -d data/
```

---

## 📁 Structure du projet

```
ecommerce-ml-analysis/
│
├── 📓 notebooks/
│   └── Projet_ML.ipynb              # Notebook principal avec toute l'analyse
│
├── 📊 data/                          # Données (non incluses dans Git)
│   ├── raw/                         # Données brutes Olist
│   │   ├── olist_orders_dataset.csv
│   │   ├── olist_order_items_dataset.csv
│   │   ├── olist_order_payments_dataset.csv
│   │   └── ...
│   └── processed/                   # Données nettoyées (générées)
│       └── cleaned_data.csv
│
├── 📈 results/                       # Résultats et 
│   └── rapport.pdf                  # Rapport académique complet
│
├── 📋 requirements.txt               # Dépendances Python
├── 📄 README.md                      # Ce fichier


```

---

## 📊 Résultats

### 🎯 Segmentation des clients (Clustering Hiérarchique)

#### Performance du modèle

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **Score de Silhouette** | 0.72 | ✅ Excellente séparation des clusters |
| **Nombre de clusters** | 3 | Segments bien distincts |
| **Méthode de linkage** | Ward | Minimisation de la variance intracluster |

#### 📌 Caractéristiques des segments

**🟦 Cluster 1 : Clients Premium (VIP)**
- 💰 Panier moyen élevé : > 200 BRL
- 🔄 Fréquence d'achat importante : 3+ commandes
- ⭐ Score de satisfaction élevé : 4.5+/5
- 🚚 Délai de livraison : Standard
- **💡 Recommandation :** Programme de fidélité, offres exclusives

**🟩 Cluster 2 : Clients Occasionnels**
- 💰 Panier moyen modéré : 80-200 BRL
- 🔄 Fréquence d'achat moyenne : 1-2 commandes
- ⭐ Score de satisfaction moyen : 3.5-4/5
- 🚚 Délai de livraison : Variable
- **💡 Recommandation :** Campagnes de réengagement, promotions ciblées

**🟥 Cluster 3 : Clients Inactifs (À risque)**
- 💰 Panier moyen faible : < 80 BRL
- 🔄 Fréquence d'achat faible : 1 commande
- ⭐ Score de satisfaction variable : 2-3.5/5
- 🚚 Délai de livraison : Long
- **💡 Recommandation :** Amélioration de l'expérience, réduction délais

### 📈 Prédiction des ventes (Régression Linéaire)

#### Performance du modèle

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **R² Score** | 0.127 | 12.7% de variance expliquée |
| **MSE** | 38,569.68 | Erreur quadratique moyenne |
| **RMSE** | 196.39 BRL | Erreur moyenne en BRL |

#### 🔍 Facteurs influençant les ventes

**Coefficients de régression :**

| Variable | Coefficient | Impact |
|----------|------------|--------|
| `credit_card` | +29.48 | ⬆️ Paiement par carte augmente les ventes |
| `delivery_time` | -8.13 | ⬇️ Délai long diminue les ventes |
| `boleto` | -73.03 | ⬇️ Paiement boleto associé à paniers plus petits |

**Observations clés :**
- 💳 Le **paiement par carte de crédit** favorise les achats plus élevés
- ⏰ Les **délais de livraison** impactent négativement les ventes
- 📦 L'optimisation logistique est cruciale pour améliorer les performances

#### 🎯 Limites identifiées

- R² modeste (0.127) : Le modèle explique une part limitée de la variance
- **Raisons possibles :**
  - Variables explicatives insuffisantes
  - Relations non-linéaires non capturées
  - Besoin d'algorithmes plus complexes (Random Forest, XGBoost)

---

## 📸 Visualisations

### 1. Distribution des types de paiement

> Analyse de la répartition des méthodes de paiement utilisées par les clients

**Insights :**
- 73.67% des transactions sont réalisées par **carte de crédit**
- Les paiements **boleto** représentent 19.46% des transactions
- Faible utilisation des paiements **debit_card** (1.43%)

### 2. Évolution temporelle des commandes

> Tendance du nombre de commandes journalières entre 2016 et 2018

**Insights :**
- Forte augmentation en **novembre 2017** (Black Friday)
- Croissance régulière du volume de commandes
- Saisonnalité visible avec des pics pendant les périodes promotionnelles

### 3. Dendrogramme du clustering hiérarchique

> Visualisation de la hiérarchie des clusters formés

**Insights :**
- Seuil de coupure optimal identifié pour 3 clusters
- Bonne séparation des groupes
- Cohérence intracluster élevée

### 4. Visualisation 2D des clusters

> Répartition des clients segmentés selon délai de livraison et temps d'approbation

**Insights :**
- 3 groupes bien distincts visuellement
- Cluster 1 (rouge) : Clients avec délais courts
- Cluster 2 (bleu) : Clients avec délais modérés
- Cluster 3 (vert) : Clients avec délais longs

### 5. Résultats de la régression

**Distribution des erreurs résiduelles :**
- Distribution asymétrique
- Concentration des erreurs près de zéro
- Quelques outliers à traiter

**Valeurs réelles vs Prédictions :**
- Corrélation visible mais imparfaite
- Sous-estimation pour les valeurs élevées
- Potentiel d'amélioration avec d'autres modèles

---

## 💡 Conclusions et perspectives

### 🎯 Conclusions principales

#### ✅ Réalisations

1. **Segmentation efficace des clients**
   - 3 segments distincts identifiés avec un score de Silhouette de 0.72
   - Profils clients clairement définis pour le marketing ciblé
   - Base solide pour la personnalisation de l'expérience client

2. **Insights actionnables pour le business**
   - Le type de paiement influence significativement les montants d'achat
   - Le délai de livraison est un facteur critique pour la satisfaction
   - Opportunités d'optimisation identifiées dans la logistique

3. **Méthodologie robuste**
   - Pipeline complet de data science appliqué
   - Traitement rigoureux des données manquantes et outliers
   - Validation des modèles avec métriques appropriées

#### ⚠️ Limitations identifiées

1. **Performance prédictive modérée**
   - R² de 0.127 indique que le modèle linéaire est insuffisant
   - Relations complexes non capturées par la régression linéaire

2. **Scope des données**
   - Dataset limité à 2016-2018 (nécessite mise à jour)
   - Marché brésilien spécifique (généralisation limitée)

### 🚀 Perspectives d'amélioration

#### Court terme (1-3 mois)

1. **🤖 Modèles avancés de prédiction**
   - Implémenter **Random Forest** et **XGBoost** pour capturer les non-linéarités
   - Tester des **réseaux de neurones** (MLP) pour la prédiction
   - Comparer les performances et sélectionner le meilleur modèle

2. **📊 Feature Engineering amélioré**
   - Créer des variables d'interaction (ex: délai × type_paiement)
   - Extraire des features temporelles (jour de la semaine, mois, saison)
   - Calculer des métriques RFM (Recency, Frequency, Monetary)

3. **🔍 Analyse approfondie des segments**
   - Profiler chaque cluster avec des statistiques descriptives détaillées
   - Analyser les catégories de produits préférées par segment
   - Étudier les patterns géographiques de chaque cluster

#### Moyen terme (3-6 mois)

4. **⏰ Analyse de séries temporelles**
   - Implémenter **ARIMA** ou **Prophet** pour la prédiction de ventes
   - Détecter les tendances saisonnières et cycles
   - Prévoir les pics de demande (Black Friday, Noël)

5. **🎯 Système de recommandation**
   - Développer un moteur de recommandation collaboratif
   - Recommandations personnalisées par segment client
   - Cross-selling et up-selling intelligents

6. **📱 Analyse de sentiment des avis**
   - NLP sur les commentaires clients (`review_comment_message`)
   - Extraction de topics récurrents
   - Corrélation sentiment/satisfaction/ventes

#### Long terme (6-12 mois)

7. **🌐 Dashboard interactif**
   - Développer une application **Streamlit** ou **Dash**
   - Visualisations interactives en temps réel
   - Interface pour les équipes marketing et commerciales

8. **🔮 Prédiction de churn**
   - Identifier les clients à risque de départ
   - Scoring de probabilité de réachat
   - Actions préventives ciblées

9. **📈 Optimisation des campagnes marketing**
   - A/B testing guidé par ML
   - Personnalisation des offres par segment
   - Optimisation du ROI marketing



---

 **Brazilian E-Commerce Public Dataset by Olist**  
   Kaggle Dataset  
   [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

### Documentation technique

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

---




**⭐ Si ce projet vous intéresse, n'hésitez pas à mettre une étoile sur GitHub ! ⭐**

