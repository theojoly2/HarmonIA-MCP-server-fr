# HarmonIA-MCP-Server

Serveur MCP pour l'assistant de modélisation sémantique **HarmonIA**. Il expose un ensemble d'outils et de ressources permettant à un agent LLM de valider, analyser, améliorer et transformer des modèles de données sémantiques.

---

## Ressources

Les ressources sont des fichiers et répertoires statiques que l'agent peut lire et exploiter.

| Ressource | Chemin | Description |
|---|---|---|
| Guide de style SEMIC (Excel) | `resources/semantic_conventions/style_guide/SEMIC_Style_Guide.xlsx` | Règles et conventions du guide SEMIC lisibles par la machine |
| Guide de style SEMIC (texte) | `resources/semantic_conventions/style_guide/style_guide.txt` | Description textuelle du guide SEMIC |
| Modèles utilisateurs | `resources/semantic_model/models/<user>/<session>/` | Modèles uploadés, organisés par utilisateur et session |

---

## Outils

Les outils sont des fonctions appelables par l'agent pour analyser et modifier les modèles.

| Outil | Description |
|---|---|
| `get_resources` | Liste les ressources et modèles disponibles pour un utilisateur et une session. |
| `index_search` | Recherche sémantique dans l'index Qdrant pour trouver des concepts standards (SEMIC, Schema.org, LOV, etc.). |
| `retrieve_document_context` | Extrait le contexte RAG le plus pertinent d'un document indexé. |
| `metadata_checker` | Vérifie la complétude des métadonnées et la cohérence terminologique du modèle. |
| `planning_orchestrator` | Génère un plan structuré pour répondre à une question utilisateur en utilisant les outils disponibles. |
| `semantic_model` | Lit et modifie le modèle de données uploadé (ajout/mise à jour de classes, attributs, connecteurs). |
| `reuse_check` | Vérifie si le modèle réutilise correctement des concepts issus de vocabulaires standards. |
| `style_guide_checks` | Agrège les résultats des différentes vérifications dans un rapport Markdown. |
| `validator_check` | Valide le modèle contre le guide SEMIC via le validateur SHACL ITB de l'UE. |

---

## Structure du projet

```
HarmonIA-MCP-server-fr/
│
├── server.py                          # Point d'entrée du serveur MCP
│
├── resources/
│   ├── semantic_conventions/
│   │   └── style_guide/
│   │       ├── SEMIC_Style_Guide.xlsx
│   │       └── style_guide.txt
│   └── semantic_model/
│       └── models/
│           └── <user>/<session>/      # Modèles uploadés
│
├── tools/
│   ├── get_resources/
│   ├── index_search/                  # Recherche vectorielle Qdrant + indexation
│   ├── model_metadata_checks/
│   ├── planning_orchestrator/
│   ├── semantic_model/                # Lecture / modification du modèle utilisateur
│   ├── semantic_reuse_of_existing_concepts_checks/
│   ├── style_guide_checks/
│   └── style_guide_validator/
│
├── config.py                          # Chargement de la configuration
├── config.yaml                        # Configuration applicative (chemins, paramètres)
├── docker-compose.yml                 # Qdrant en local
├── .env                               # Variables d'environnement (non commité)
├── .env.sample                        # Exemple de variables d'environnement
└── requirements.txt
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone <url-du-depot>
cd HarmonIA-MCP-server-fr
```

### 2. Créer l'environnement virtuel

```bash
python -m venv venv-server
source venv-server/bin/activate  # Windows : venv-server\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

```bash
cp .env.sample .env
# Éditer .env avec vos clés API et endpoints
```

Variables principales :

| Variable | Description |
|---|---|
| `SERVER_HOST` / `SERVER_PORT` | Hôte et port de Qdrant (par défaut `qdrant:6333` en Docker) |
| `EMBEDDING_API_KEY` / `URL_EMBEDDING_API` | API d'embeddings compatible OpenAI (ex. Albert) |
| `RERANKER_API_KEY` / `URL_RERANKER_API` | API de reranking compatible OpenAI (ex. Albert) |
| `LLM_API_KEY` / `URL_LLM_API` / `LLM_MODEL` | API LLM pour la génération de résumés (optionnel) |

> Par défaut, l'indexation et le reranking peuvent aussi s'effectuer **localement** avec BGE-M3 et BGE-Reranker-V2-M3 si aucune API externe n'est configurée.

---

## Indexation des documents

### 1. Démarrer Qdrant

```bash
docker-compose up -d
```

Qdrant est accessible sur `http://localhost:6333`.

### 2. Placer les documents à indexer

```bash
mkdir -p tools/index_search/load_documents/documents
```

Ajouter les fichiers dans `documents/`. L'arborescence des sous-dossiers détermine les tags de filtrage. Chaque niveau de dossier devient un tag attaché au fichier ; un fichier placé dans plusieurs sous-dossiers hérite de tous les tags de son chemin.

Exemple :

```
documents/schema.data.gouv.fr/
documents/SEMIC/
documents/schema.org/
documents/schema.data.gouv.fr/CNIG/mon_schema.ttl
```

Dans le dernier cas, `mon_schema.ttl` reçoit les deux tags `schema.data.gouv.fr` et `CNIG`. L'interface et l'API permettent ensuite de filtrer les recherches par un ou plusieurs de ces tags.

### 3. Lancer l'indexation

```bash
python tools/index_search/load_documents/load.py
```

La collection `documents_collection` est créée automatiquement avec un vecteur dense nommé `dense` et les indexes de payload `tags`, `document_id` et `summary_enabled`.

### Corpus de documents pré-construit

Une branche Git dédiée contient un corpus de référence prêt à être indexé : **`corpus-preindexe`**. Cette branche inclut les fichiers sources dans `tools/index_search/load_documents/documents/` (schema.data.gouv.fr, schema.org, FIWARE, INSPIRE, LOV, SAREF, SEMIC, CNIG), mais pas l'index Qdrant lui-même. Pour l'utiliser :

```bash
git fetch origin
git checkout corpus-preindexe
python tools/index_search/load_documents/load.py
```

Une fois indexé, revenir sur la branche principale pour continuer le développement. L'index Qdrant persiste dans `docker-compose` / `data/qdrant` et reste utilisable.

### 4. (Optionnel) Tester la recherche

```bash
python tools/index_search/retrieve_search_documents.py
```

---

## Lancement du serveur

```bash
python -m server
```

Le serveur MCP écoute sur `0.0.0.0:8001` par défaut.

---

## Exécution complète (en dev)

**Important** : l'indexation doit être réalisée **une seule fois** avant la première utilisation. Elle n'est pas à relancer à chaque démarrage, seulement si les documents sources changent. Tant que Qdrant est vide, la recherche documentaire ne retournera aucun résultat.

Ordre de lancement :

```bash
# 1. Démarrer Qdrant
docker-compose up -d

# 2. Indexer les documents (une seule fois au premier lancement)
source venv-server/bin/activate
python tools/index_search/load_documents/load.py

# 3. Lancer le serveur MCP
source venv-server/bin/activate
python -m server
```

Chaque commande peut s'exécuter dans son propre terminal. Une fois Qdrant démarré et les documents indexés, il suffit de relancer uniquement le serveur MCP lors des démarrages suivants.

---

## Sécurité

- Ne jamais commiter `.env` ni les clés API.
- Les modèles utilisateurs sont stockés en clair sur disque : activer le chiffrement au repos en production.
- Limiter l'accès réseau au serveur MCP et à Qdrant.
