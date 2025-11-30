# AiOPs Log Anomaly Detection (HDFS & BGL)

## Description

Ce projet implémente un pipeline complet d’AiOPs pour la détection d’anomalies à partir de logs système, dans le cadre du laboratoire du cours MGL870 — Observabilité & Génie logiciel.

Il couvre l’ensemble des étapes d’un pipeline moderne :
1. Parsing des logs bruts (Drain)
2. Extraction des templates & événements
3.	Construction des séquences (HDFS → block_id, BGL → node_id)
4.	Transformation en features (matrice événements × fréquence)
5.	Entraînement de modèles ML supervisés:
- Logistic Regression
- Random Forest
6.	Analyse des performances (Precision, Recall, AUC)
7.	Interprétation du modèle (features importantes)

### Structure du projet
```
aiops-log-lab/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── parse_hdfs.py
│   ├── parse_bgl.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_sequences.py
│   └── models/
│       ├── __init__.py
│       └── train_models.py
├── data/
│   ├── raw/
│   │   ├── hdfs/        ← logs HDFS_v1
│   │   └── bgl/         ← logs BGL
│   └── parsed/
│       ├── hdfs/        ← sorties Drain (HDFS)
│       └── bgl/         ← sorties Drain (BGL)
├── reports/
│   ├── figures/
│   └── notes.md
└── .gitignore
```

### Datasets

Les fichiers de logs ne sont pas inclus dans ce dépôt en raison de leur taille.
Ils doivent être téléchargés depuis LogHub : https://github.com/logpai/loghub?tab=readme-ov-file
- HDFS_v1: https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1
- BGL: https://zenodo.org/records/8196385/files/BGL.zip?download=1

Place les fichiers dans :
```
data/raw/hdfs/
data/raw/bgl/
```

Par exemple :
```
data/raw/hdfs/HDFS.log
data/raw/bgl/BGL.log
```


## Installation

Créer l’environnement et installer les dépendances :
```
pip install -r requirements.txt
```

## Parsing des logs (Drain)

HDFS
```
python src/parse_hdfs.py
```

BGL
```
python src/parse_bgl.py
```

Les sorties seront générées dans :
```
data/parsed/hdfs/
data/parsed/bgl/
```

Chaque dataset générera :
- *_structured.csv → logs structurés + EventId
- templates.csv → table des templates

## Construction des séquences

Exécuter :
```
python src/features/build_sequences.py
```

Ce script :
- regroupe HDFS par block_id
- regroupe BGL par node_id
- génère la matrice d’événements utilisable par les modèles ML

## Entraînement des modèles ML
```
python src/models/train_models.py
```

Ce script :
- applique un split temporel
- calcule la corrélation & supprime les variables fortement corrélées
- entraîne Logistic Regression & Random Forest
- génère les métriques : Precision, Recall, AUC
- identifie les features importantes (permutation importance)

Les résultats et graphs seront enregistrés dans reports/figures/.

## Résultats attendus

Le pipeline permet de :
- détecter automatiquement les anomalies présentes dans les traces HDFS et BGL
- comparer deux algorithmes supervisés simples
- comprendre quels événements sont les plus discriminants
- reproduire exactement les étapes vues dans le cours (parsing → ML → interprétation)