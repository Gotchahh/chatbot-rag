# Chatbot RAG su PDF

Un chatbot basato su **Retrieval-Augmented Generation (RAG)** che permette di caricare un documento PDF e fare domande sul suo contenuto. Il sistema estrae il testo, lo indicizza in un database vettoriale e utilizza un LLM per generare risposte basate esclusivamente sul documento caricato.

## Come funziona

Il progetto si basa su due flussi principali:

**Indicizzazione del documento:** quando l'utente carica un PDF, il sistema estrae il testo, lo divide in chunk (frammenti) di 600 caratteri con un overlap di 150, genera gli embeddings tramite HuggingFace e li salva in ChromaDB.

**Generazione della risposta:** quando l'utente fa una domanda, il sistema genera l'embedding della domanda, cerca i 10 chunk più simili nel database vettoriale, li passa come contesto al modello LLM (Llama 3.3 70B tramite Groq) che genera una risposta basata esclusivamente sul contenuto del documento.

```
PDF → Estrazione testo → Chunking → Embeddings → ChromaDB
                                                       ↓
Domanda utente → Embedding → Ricerca similarità → Contesto
                                                       ↓
                                                LLM → Risposta
```

## Stack tecnologico

- **Python** — linguaggio principale
- **Streamlit** — interfaccia web
- **LangChain** — orchestrazione del flusso RAG
- **ChromaDB** — database vettoriale per la ricerca semantica
- **HuggingFace (all-MiniLM-L6-v2)** — modello di embedding per la rappresentazione vettoriale del testo
- **Groq (Llama 3.3 70B)** — LLM per la generazione delle risposte
- **PyMuPDF** — estrazione testo dai PDF
- **Docker** — containerizzazione dell'applicazione

## Prerequisiti

- Python 3.9+ oppure Docker
- Una chiave API gratuita di [Groq](https://console.groq.com)

## Installazione

### Con Docker (consigliato)

```bash
git clone https://github.com/Gotchahh/chatbot-rag.git
cd chatbot-rag
```

Crea il file `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "la-tua-chiave-groq"
```

Builda e avvia:

```bash
docker build -t chatbot-rag .
docker run -p 8501:8501 chatbot-rag
```

Apri il browser su `http://localhost:8501`.

### Con venv (sviluppo locale)

```bash
git clone https://github.com/Gotchahh/chatbot-rag.git
cd chatbot-rag
python3 -m venv venv
source venv/bin/activate  # su Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Crea il file `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "la-tua-chiave-groq"
```

Avvia:

```bash
streamlit run app.py
```

## Struttura del progetto

```
chatbot-rag/
├── app.py                  # Applicazione principale
├── Dockerfile              # Configurazione Docker
├── requirements.txt        # Dipendenze Python
├── .streamlit/
│   └── secrets.toml        # Chiavi API (non versionato)
├── .gitignore
└── README.md
```

## Parametri configurabili

| Parametro | Valore | Descrizione |
|---|---|---|
| chunk_size | 600 | Dimensione di ogni frammento di testo in caratteri |
| chunk_overlap | 150 | Sovrapposizione tra chunk consecutivi per non perdere informazioni nei tagli |
| k | 10 | Numero di chunk recuperati per ogni domanda |
| model | llama-3.3-70b-versatile | Modello LLM utilizzato tramite Groq |
| embedding model | all-MiniLM-L6-v2 | Modello di embedding HuggingFace |

## Possibili miglioramenti

- Implementazione di **reranking** per riordinare i chunk recuperati per rilevanza
- Utilizzo di un **modello di embedding multilingua** (es. `intfloat/multilingual-e5-large`) per migliorare la precisione su documenti in italiano
- Aggiunta di **ricerca ibrida** (semantica + keyword) tramite MMR (Maximum Marginal Relevance)
- Implementazione di **memoria della conversazione** per domande di follow-up
- Deploy su **AWS** o **Google Cloud**

## Autore

Realizzato come progetto formativo nell'ambito del percorso ITS AI Developing.
