#  Assistant IA & Analyse de Documents
> Chatbot RAG (Retrieval-Augmented Generation) local avec DeepSeek R1, ChromaDB et Streamlit

---

##  C'est quoi ce projet ?

Un chatbot intelligent qui **lit tes documents PDF ou images** et répond à tes questions en français.  
Il tourne entièrement en local sur ta machine — aucune donnée ne part sur internet.

**Comment ça marche :**
1. Tu uploades un PDF ou une image
2. Le texte est découpé en morceaux (chunks) et converti en vecteurs (embeddings)
3. Quand tu poses une question, les passages les plus pertinents sont retrouvés
4. DeepSeek R1 génère une réponse basée uniquement sur ces passages

---

##  Structure du projet

```
projet/
├── app.py                  # Interface Streamlit (chatbot avec historique)
├── src/
│   ├── brain.py            # Logique IA : embeddings, vectorDB, LLM
│   └── processor.py        # Extraction de texte (PDF + OCR image)
├── data/
│   ├── uploads/            # Fichiers uploadés par l'utilisateur
│   └── chromadb/           # Base vectorielle (générée automatiquement)
└── requirements.txt
```

---

##  Prérequis

- Python 3.10+
- [Ollama](https://ollama.com) installé sur la machine
- poppler (pour la lecture PDF scannés)

### Installer poppler (Fedora/Linux)
```bash
sudo dnf install poppler-utils
```

### Installer poppler (Ubuntu/Debian)
```bash
sudo apt install poppler-utils
```

---

## 🚀 Installation

### 1. Cloner le projet
```bash
git clone <url-du-repo>
cd <nom-du-dossier>
```

### 2. Créer l'environnement virtuel
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances Python
```bash
pip install -r requirements.txt
```

### 4. Installer le modèle DeepSeek via Ollama

>  **Recommandé pour la vitesse**

| Modèle | RAM nécessaire | Vitesse | Qualité |
|--------|---------------|---------|---------|
| `deepseek-r1:1.5b` | ~2 Go | ⚡⚡⚡ Rapide | Correct |
| `deepseek-r1:7b` | ~8 Go |  Lent sans GPU | Bon |
| `llama3.2:3b` | ~3 Go | ⚡⚡⚡ Très rapide | Bon |

```bash
# Option rapide (recommandée)
ollama pull deepseek-r1:1.5b

# Ou version plus puissante (nécessite un bon GPU)
ollama pull deepseek-r1:7b
```

Ensuite dans `src/brain.py`, ajuste la ligne :
```python
model="deepseek-r1:1.5b"   # change selon ce que tu as installé
```

---

##  Lancer l'application

```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Lancer Streamlit
streamlit run app.py
```

L'application s'ouvre sur **http://localhost:8501**

---

##  Optimisations appliquées

Ce projet a été optimisé pour être le plus rapide possible sur une machine sans GPU :

| Optimisation | Détail |
|---|---|
| **Singleton embeddings** | Le modèle d'embedding est chargé une seule fois en mémoire |
| **Singleton LLM** | DeepSeek est chargé une seule fois, pas rechargé à chaque question |
| **Cache vectorDB** | Un fichier déjà uploadé n'est pas re-vectorisé |
| **`@st.cache_resource`** | Streamlit ne recharge pas les modèles à chaque interaction |
| **`session_state`** | La base de données et l'historique survivent aux reruns Streamlit |
| **k=5 chunks max** | Seulement 5 passages envoyés au LLM (au lieu de 20) |
| **`num_predict=800`** | Limite la longueur de réponse pour éviter les timeouts |

---

##  Fichiers clés expliqués

### `src/brain.py`
- `get_embeddings()` : charge le modèle `all-MiniLM-L6-v2` (transforme le texte en vecteurs)
- `get_llm()` : charge DeepSeek R1 via Ollama
- `split_text()` : découpe le texte en morceaux de 800 caractères
- `save_to_vector_db()` : stocke les vecteurs dans ChromaDB
- `answer_with_deepseek()` : génère la réponse à partir des passages pertinents
- `clear_memory()` : efface la base de données et remet les modèles à zéro

### `src/processor.py`
- Lit les PDF avec `pypdf` (texte natif)
- Si le PDF est scanné, utilise `EasyOCR` pour lire les images
- Supporte aussi les images JPG/PNG directement

### `app.py`
- Interface Streamlit avec historique de chat
- Upload de documents dans la sidebar
- Affichage des passages sources utilisés pour la réponse

---

##  Problèmes fréquents

**La réponse prend plus de 5 minutes**
→ Ton modèle est trop lourd pour ta machine. Passe à `deepseek-r1:1.5b` ou `llama3.2:3b`

**La réponse est vide**
→ DeepSeek R1 produit des balises `<think>` supprimées par le nettoyage. Vérifie le bloc Debug dans l'interface.

**Erreur poppler**
→ Installe `poppler-utils` (voir prérequis)

**Erreur ChromaDB "read-only"**
→ Clique sur "Réinitialiser les documents" dans la sidebar, puis recharge le fichier

---

##  Dépendances principales

```
streamlit
langchain
langchain-huggingface
langchain-chroma
langchain-ollama
sentence-transformers
chromadb
pypdf
easyocr
pdf2image
numpy
```

---

*Projet développé et optimisé avec assistance Claude (Anthropic)*
